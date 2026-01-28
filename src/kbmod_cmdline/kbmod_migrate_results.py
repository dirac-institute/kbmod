"""A program to migrate legacy KBMOD results parquet files to the new format
with proper image column metadata and auxiliary files.

This tool:
- Reshapes flattened image columns back to 2D (stamp_dim x stamp_dim)
- Writes image columns as separate auxiliary FITS files
- Validates migration by comparing row/column counts
- Replaces original files with migrated versions on success
- Logs all operations to CSV files for audit trail

Basic usage to migrate files with coadd columns:

>>> kbmod-migrate-results --input=/path/to/results.parquet \\
...     --image-columns '*coadd*' --stamp-dim 101

To migrate all parquet files in a directory:

>>> kbmod-migrate-results --input=/path/to/results_dir \\
...     --image-columns '*coadd*' 'stamps' --stamp-dim 101

Dry run to preview what would be migrated:

>>> kbmod-migrate-results --input=/path/to/results_dir \\
...     --image-columns '*coadd*' --stamp-dim 101 --dry-run -v

Parallel processing with 8 workers:

>>> kbmod-migrate-results --input=/path/to/results_dir \\
...     --image-columns '*coadd*' --stamp-dim 101 --workers 8

Chunked reading for large files (note that this ignores existing auxiliary files):

>>> kbmod-migrate-results --input=/path/to/large_results.parquet \\
...     --image-columns '*coadd*' --stamp-dim 101 --chunk-size 10000
"""

import argparse
import csv
import fnmatch
import logging
import os
import shutil
import tempfile
from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from kbmod.results import Results

logger = logging.getLogger(__name__)

# Result of processing a single file
MigrationResult = namedtuple(
    "MigrationResult",
    [
        "success",  # bool: True if migration succeeded
        "original_path",  # Path: original file path
        "final_path",  # Path: final file path (same as original if replaced)
        "aux_files",  # list[str]: names of auxiliary files created
        "rows",  # int: number of rows migrated
        "skipped",  # bool: True if file was skipped
        "skip_reason",  # str: reason for skipping (if skipped)
        "error_msg",  # str: error message (if failed)
    ],
)


def find_parquet_files(input_path, glob_pattern="**/*.search.parquet"):
    """Find all parquet files matching the criteria.

    Parameters
    ----------
    input_path : `str` or `Path`
        Path to a single file or directory to search.
    glob_pattern : `str`
        Glob pattern for finding parquet files in a directory.

    Returns
    -------
    files : `list[Path]`
        List of parquet file paths.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        # Check for compound extensions using name (Path.suffix only returns last extension)
        name_lower = input_path.name.lower()
        if name_lower.endswith(".search.parquet") or name_lower.endswith(".search.parq"):
            return [input_path]
        else:
            raise ValueError(f"Input file must be a .search.parquet file: {input_path}")

    if input_path.is_dir():
        files = list(input_path.glob(glob_pattern))
        # Also check for .parq extension if pattern uses .parquet
        if ".search.parquet" in glob_pattern:
            parq_pattern = glob_pattern.replace(".search.parquet", ".search.parq")
            files.extend(input_path.glob(parq_pattern))
        return sorted(set(files))

    raise FileNotFoundError(f"Input path not found: {input_path}")


def match_column_patterns(colnames, patterns):
    """Match column names against glob patterns.

    Parameters
    ----------
    colnames : `list[str]`
        List of column names to match against.
    patterns : `list[str]`
        List of patterns (supports wildcards like '*coadd*', 'stamp*').

    Returns
    -------
    matched : `list[str]`
        List of column names that match any pattern.

    Examples
    --------
    >>> match_column_patterns(['coadd_sum', 'coadd_mean', 'flux'], ['*coadd*'])
    ['coadd_sum', 'coadd_mean']
    >>> match_column_patterns(['stamps', 'stamp_data', 'flux'], ['stamp*'])
    ['stamps', 'stamp_data']
    """
    matched = set()
    for pattern in patterns:
        for col in colnames:
            if fnmatch.fnmatch(col, pattern):
                matched.add(col)
    return list(matched)


def find_auxiliary_files(base_path):
    """Find existing auxiliary files for a result file.

    Parameters
    ----------
    base_path : `Path`
        Path to the main results parquet file.

    Returns
    -------
    aux_files : `list[tuple[str, Path]]`
        List of (column_name, aux_path) tuples.
    """
    base_path = Path(base_path)
    parent = base_path.parent
    stem = base_path.stem

    aux_files = []

    # Look for auxiliary files matching pattern: {stem}_{colname}.fits or .parquet
    for ext in [".fits", ".parquet", ".parq"]:
        for aux_path in parent.glob(f"{stem}_*{ext}"):
            # Extract column name from filename
            # e.g., "results_coadd_sum.fits" -> "coadd_sum"
            aux_stem = aux_path.stem
            if aux_stem.startswith(stem + "_"):
                col_name = aux_stem[len(stem) + 1 :]
                aux_files.append((col_name, aux_path))

    return aux_files


def has_auxiliary_files(base_path):
    """Check if a result file already has auxiliary files.

    Parameters
    ----------
    base_path : `Path`
        Path to the main results parquet file.

    Returns
    -------
    has_aux : `bool`
        True if auxiliary files exist.
    """
    return len(find_auxiliary_files(base_path)) > 0


def count_parquet_rows_and_columns(filepath):
    """Count rows and columns in a parquet file without loading data.

    Parameters
    ----------
    filepath : `Path`
        Path to the parquet file.

    Returns
    -------
    num_rows : `int`
        Number of rows in the file.
    num_cols : `int`
        Number of columns in the file.
    colnames : `list[str]`
        List of column names.
    """
    pf = pq.ParquetFile(filepath)
    num_rows = pf.metadata.num_rows
    colnames = pf.schema_arrow.names
    return num_rows, len(colnames), colnames


def validate_migration(original_path, new_path, expected_aux_files, expected_image_shapes=None):
    """Validate that migration was successful.

    Parameters
    ----------
    original_path : `Path`
        Path to the original parquet file.
    new_path : `Path`
        Path to the new parquet file.
    expected_aux_files : `list[Path]`
        List of expected auxiliary file paths.
    expected_image_shapes : `dict`, optional
        Expected image_column_shapes metadata. If provided, validates that
        the new parquet file contains this metadata correctly.

    Returns
    -------
    valid : `bool`
        True if validation passes.
    error_msg : `str`
        Error message if validation fails, empty string otherwise.
    """
    try:
        # Check new file exists
        if not new_path.exists():
            return False, f"New file does not exist: {new_path}"

        # Check all auxiliary files exist
        for aux_path in expected_aux_files:
            if not aux_path.exists():
                return False, f"Auxiliary file does not exist: {aux_path}"

        if len(expected_aux_files) != len(expected_image_shapes):
            return (
                False,
                f"Mismatch in expected auxiliary files and expected image columns {len(expected_image_shapes)} vs {len(expected_aux_files)}",
            )

        # Compare row counts
        orig_rows, orig_cols, _ = count_parquet_rows_and_columns(original_path)
        new_rows, new_cols, _ = count_parquet_rows_and_columns(new_path)

        if orig_rows != new_rows:
            return False, f"Row count mismatch: original={orig_rows}, new={new_rows}"

        # Column count: original should equal new + number of aux files
        # (since aux columns are removed from main file)
        expected_new_cols = orig_cols - len(expected_aux_files)
        if new_cols != expected_new_cols:
            return (
                False,
                f"Column count mismatch: expected {expected_new_cols} "
                f"(orig={orig_cols} - aux={len(expected_aux_files)}), got {new_cols}",
            )

        # Validate image_column_shapes metadata if expected shapes provided
        if expected_image_shapes:
            pf = pq.ParquetFile(new_path)
            meta_dict = Results._extract_parquet_metadata(pf)
            stored_shapes = meta_dict.get("image_column_shapes", {})

            # Check that all expected image columns are in metadata
            for col, expected_shape in expected_image_shapes.items():
                if col not in stored_shapes:
                    return False, f"Missing image_column_shapes metadata for column '{col}'"

                stored_shape = (
                    tuple(stored_shapes[col]) if isinstance(stored_shapes[col], list) else stored_shapes[col]
                )
                expected_shape = tuple(expected_shape) if isinstance(expected_shape, list) else expected_shape

                if stored_shape != expected_shape:
                    return (
                        False,
                        f"Shape mismatch for column '{col}': "
                        f"expected {expected_shape}, got {stored_shape}",
                    )

        return True, ""

    except Exception as e:
        return False, f"Validation error: {e}"


def move_files_to_original_location(temp_base, temp_aux_files, original_path):
    """Move temp files to replace the original.

    Parameters
    ----------
    temp_base : `Path`
        Path to the temporary base parquet file.
    temp_aux_files : `list[Path]`
        List of temporary auxiliary file paths.
    original_path : `Path`
        Path to the original file to replace.

    Returns
    -------
    final_paths : `list[Path]`
        List of final file paths (base + aux files).
    """
    original_path = Path(original_path)
    original_dir = original_path.parent

    final_paths = []

    # shutil.move handles overwriting - atomic on same filesystem
    shutil.move(str(temp_base), str(original_path))
    final_paths.append(original_path)

    # Move auxiliary files to original directory
    for aux_path in temp_aux_files:
        final_aux = original_dir / aux_path.name
        shutil.move(str(aux_path), str(final_aux))
        final_paths.append(final_aux)

    return final_paths


def reshape_image_columns_inplace(results, matched_columns, stamp_dim):
    """Reshape image columns from flattened 1D to 2D stamps.

    Parameters
    ----------
    results : `Results`
        The Results object to modify in place.
    matched_columns : `list[str]`
        List of column names to reshape.
    stamp_dim : `int`
        The stamp dimension (stamps are stamp_dim x stamp_dim).

    Raises
    ------
    ValueError
        If a column entry has wrong size or is not a numpy array.
    """
    shape_2d = (stamp_dim, stamp_dim)
    expected_size = stamp_dim * stamp_dim

    for col in matched_columns:
        if col not in results.colnames:
            raise ValueError(f"Column {col} not found in results")

        reshaped_data = []
        for entry in results.table[col]:
            if isinstance(entry, np.ndarray):
                if entry.size == expected_size:
                    reshaped_data.append(entry.reshape(shape_2d))
                else:
                    raise ValueError(f"Column {col}: entry size {entry.size} != expected {expected_size}")
            else:
                raise ValueError(f"Column {col}: entry is not a numpy.ndarray")

        results.table[col] = reshaped_data


def load_and_reshape_results(file_path, matched_columns, stamp_dim, chunk_size=None):
    """Load results file and reshape image columns.

    For large files (or when use_chunking=True), reads in chunks to manage memory if
    `chunk_size` is specified. Note that chunked reading ignores existing auxiliary files.

    Parameters
    ----------
    file_path : `Path`
        Path to the parquet file.
    matched_columns : `list[str]`
        List of image column names to reshape.
    stamp_dim : `int`
        The stamp dimension.
    chunk_size : `int`
        Number of rows per chunk when using chunked reading. If None, reads entire file at once.

    Returns
    -------
    results : `Results`
        The loaded and reshaped Results object.
    num_rows : `int`
        Total number of rows loaded.
    """
    if chunk_size is not None:
        # Use chunked reading for memory efficiency
        accumulated_results = None
        num_rows = 0

        # Note that read_table_chunks ignores existing auxiliary files. However, if migrating
        # from legacy format, auxiliary files for image columns should not exist yet.
        for chunk in Results.read_table_chunks(str(file_path), chunk_size=chunk_size):
            # Reshape image columns in this chunk
            reshape_image_columns_inplace(chunk, matched_columns, stamp_dim)
            num_rows += len(chunk)

            if accumulated_results is None:
                accumulated_results = chunk
            else:
                accumulated_results.extend(chunk)

        if accumulated_results is None:
            # Empty file
            accumulated_results = Results.from_trajectories([])
            num_rows = 0

        return accumulated_results, num_rows
    else:
        # Load entire file at once (for smaller files)
        results = Results.read_table(str(file_path))
        num_rows = len(results)
        reshape_image_columns_inplace(results, matched_columns, stamp_dim)
        return results, num_rows


def process_single_file(args_tuple):
    """Process a single file for migration.

    This function is designed to be used with multiprocessing.Pool.

    Parameters
    ----------
    args_tuple : `tuple`
        Tuple of (file_path, image_patterns, stamp_dim, output_dir,
                  dry_run, skip_with_aux, keep_originals, chunk_size)

    Returns
    -------
    result : `MigrationResult`
        Result of the migration attempt.
    """
    (
        file_path,
        image_patterns,
        stamp_dim,
        output_dir,
        dry_run,
        skip_with_aux,
        keep_originals,
        chunk_size,
    ) = args_tuple

    file_path = Path(file_path)

    try:
        # Check if we should skip files with existing auxiliary files
        if skip_with_aux and has_auxiliary_files(file_path):
            return MigrationResult(
                success=True,
                original_path=file_path,
                final_path=file_path,
                aux_files=[],
                rows=0,
                skipped=True,
                skip_reason="has_auxiliary_files",
                error_msg="",
            )

        # Get column names without loading full data
        _, _, colnames = count_parquet_rows_and_columns(file_path)

        # Match image column patterns
        matched_columns = match_column_patterns(colnames, image_patterns)

        if not matched_columns:
            return MigrationResult(
                success=True,
                original_path=file_path,
                final_path=file_path,
                aux_files=[],
                rows=0,
                skipped=True,
                skip_reason="no_matching_columns",
                error_msg="",
            )

        logger.info(f"Processing {file_path} with image columns: {matched_columns}")

        # Load the results file using chunked reading for memory efficiency
        results, num_rows = load_and_reshape_results(
            file_path,
            matched_columns,
            stamp_dim,
            chunk_size,
        )

        if dry_run:
            return MigrationResult(
                success=True,
                original_path=file_path,
                final_path=file_path,
                aux_files=matched_columns,
                rows=num_rows,
                skipped=False,
                skip_reason="",
                error_msg="",
            )

        # Create temp file path
        temp_dir = Path(output_dir)
        temp_base = temp_dir / file_path.name

        # Write to temp location with auxiliary files
        from kbmod.results import write_results_to_files_destructive

        write_results_to_files_destructive(
            temp_base,
            results,
            separate_col_files=matched_columns,
            image_columns=matched_columns,
            overwrite=True,
        )

        # Find the auxiliary files that were created
        temp_aux_files = []
        for col in matched_columns:
            # Image columns are saved as FITS
            aux_path = temp_dir / f"{temp_base.stem}_{col}.fits"
            if aux_path.exists():
                temp_aux_files.append(aux_path)
            else:
                # Maybe saved as parquet (non-image)
                aux_path = temp_dir / f"{temp_base.stem}_{col}.parquet"
                if aux_path.exists():
                    temp_aux_files.append(aux_path)

        # Build expected image shapes for validation
        # All matched columns should have shape (stamp_dim, stamp_dim)
        expected_image_shapes = {col: (stamp_dim, stamp_dim) for col in matched_columns}

        # Validate the migration (including metadata)
        valid, error_msg = validate_migration(file_path, temp_base, temp_aux_files, expected_image_shapes)

        if not valid:
            # Clean up temp files on validation failure
            if temp_base.exists():
                os.remove(temp_base)
            for aux in temp_aux_files:
                if aux.exists():
                    os.remove(aux)

            return MigrationResult(
                success=False,
                original_path=file_path,
                final_path=None,
                aux_files=[],
                rows=num_rows,
                skipped=False,
                skip_reason="",
                error_msg=f"Validation failed: {error_msg}",
            )

        # Move files to replace original (unless keep_originals is set)
        if keep_originals:
            final_path = temp_base
            final_aux = temp_aux_files
        else:
            final_paths = move_files_to_original_location(temp_base, temp_aux_files, file_path)
            final_path = final_paths[0]
            final_aux = [p.name for p in final_paths[1:]]

        return MigrationResult(
            success=True,
            original_path=file_path,
            final_path=final_path,
            aux_files=[p if isinstance(p, str) else p.name for p in final_aux],
            rows=num_rows,
            skipped=False,
            skip_reason="",
            error_msg="",
        )

    except Exception as e:
        logger.exception(f"Error processing {file_path}")
        return MigrationResult(
            success=False,
            original_path=file_path,
            final_path=None,
            aux_files=[],
            rows=0,
            skipped=False,
            skip_reason="",
            error_msg=str(e),
        )


def write_csv_header(filepath, headers):
    """Write CSV header if file doesn't exist.

    Parameters
    ----------
    filepath : `Path`
        Path to the CSV file.
    headers : `list[str]`
        List of column headers.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def append_csv_row(filepath, row):
    """Append a row to a CSV file.

    Parameters
    ----------
    filepath : `Path`
        Path to the CSV file.
    row : `list`
        Row data to append.
    """
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def execute(args):
    """Run the migration from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        print("KBMOD Results Migration:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Find all input files
    input_files = find_parquet_files(args.input, args.glob)
    logger.info(f"Found {len(input_files)} parquet files to process")

    if len(input_files) == 0:
        print("No parquet files found matching criteria.")
        return

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="kbmod_migrate_"))
    logger.info(f"Using output directory: {output_dir}")

    # Set up CSV files
    mapping_file = Path(args.mapping_file) if args.mapping_file else output_dir / "migration_mapping.csv"
    error_file = Path(args.error_file) if args.error_file else output_dir / "migration_errors.csv"

    mapping_headers = ["timestamp", "original_path", "final_path", "aux_files", "rows", "status"]
    error_headers = ["timestamp", "original_path", "error_type", "error_message"]

    write_csv_header(mapping_file, mapping_headers)
    write_csv_header(error_file, error_headers)

    # Prepare arguments for processing
    process_args = [
        (
            f,
            args.image_columns,
            args.stamp_dim,
            str(output_dir),
            args.dry_run,
            not args.no_skip_with_aux,  # skip_with_aux
            args.keep_originals,
            args.chunk_size,
        )
        for f in input_files
    ]

    # Process files
    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_file, process_args),
                total=len(input_files),
                desc="Migrating",
                disable=args.verbose,  # Disable tqdm in verbose mode (too much output)
            ):
                results.append(result)
                _record_result(result, mapping_file, error_file)
    else:
        for process_arg in tqdm(process_args, desc="Migrating", disable=args.verbose):
            result = process_single_file(process_arg)
            results.append(result)
            _record_result(result, mapping_file, error_file)

    # Print summary
    _print_summary(results, mapping_file, error_file, args.dry_run)


def _record_result(result, mapping_file, error_file):
    """Record a migration result to the appropriate CSV file.

    Parameters
    ----------
    result : `MigrationResult`
        The migration result.
    mapping_file : `Path`
        Path to the mapping CSV file.
    error_file : `Path`
        Path to the error CSV file.
    """
    timestamp = datetime.now().isoformat()

    if result.success:
        if result.skipped:
            status = f"skipped_{result.skip_reason}"
        else:
            status = "migrated"

        append_csv_row(
            mapping_file,
            [
                timestamp,
                str(result.original_path),
                str(result.final_path) if result.final_path else "",
                ";".join(result.aux_files),
                result.rows,
                status,
            ],
        )
    else:
        append_csv_row(
            error_file,
            [
                timestamp,
                str(result.original_path),
                "MigrationError",
                result.error_msg,
            ],
        )


def _print_summary(results, mapping_file, error_file, dry_run):
    """Print a summary of migration results.

    Parameters
    ----------
    results : `list[MigrationResult]`
        List of migration results.
    mapping_file : `Path`
        Path to the mapping CSV file.
    error_file : `Path`
        Path to the error CSV file.
    dry_run : `bool`
        Whether this was a dry run.
    """
    total = len(results)
    successful = sum(1 for r in results if r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.success)
    total_rows = sum(r.rows for r in results if r.success and not r.skipped)

    print("\n" + "=" * 50)
    if dry_run:
        print("DRY RUN SUMMARY (no files were modified)")
    else:
        print("MIGRATION SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {total}")
    print(f"  Successfully migrated: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"Total rows migrated: {total_rows}")
    print(f"\nMapping log: {mapping_file}")
    print(f"Error log: {error_file}")

    if skipped > 0:
        skip_reasons = {}
        for r in results:
            if r.skipped:
                reason = r.skip_reason
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        print("\nSkip reasons:")
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count}")

    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  {r.original_path}: {r.error_msg[:100]}")


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-migrate-results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Migrate legacy KBMOD results parquet files to the new format "
            "with proper image column metadata and auxiliary files."
        ),
    )

    parser.add_argument(
        "--input",
        dest="input",
        type=str,
        required=True,
        help="Path to a single results file or a directory to recursively search.",
    )
    parser.add_argument(
        "--image-columns",
        dest="image_columns",
        nargs="+",
        required=True,
        help=(
            "Column name patterns for image data (supports wildcards). "
            "Examples: '*coadd*' 'stamps' 'stamp*'"
        ),
    )
    parser.add_argument(
        "--stamp-dim",
        dest="stamp_dim",
        type=int,
        required=True,
        help="Stamp dimension (e.g., 101 for 101x101 stamps).",
    )
    parser.add_argument(
        "--glob",
        dest="glob",
        type=str,
        default="**/*.search.parquet",
        help="Glob pattern for finding parquet files in a directory.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help=(
            "Base directory for staging migrated files. "
            "Files are moved to original location on success unless --keep-originals is set."
        ),
    )
    parser.add_argument(
        "--mapping-file",
        dest="mapping_file",
        type=str,
        default=None,
        help="Path to the migration mapping CSV file.",
    )
    parser.add_argument(
        "--error-file",
        dest="error_file",
        type=str,
        default=None,
        help="Path to the error log CSV file.",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=None,
        help="Chunk size for reading large files. Ignores existing auxiliary files when set.",
    )

    optional = parser.add_argument_group("Optional flags")
    optional.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Don't write files, just report what would be done.",
    )
    optional.add_argument(
        "--no-skip-with-aux",
        dest="no_skip_with_aux",
        action="store_true",
        default=False,
        help="Process files even if they already have auxiliary files.",
    )
    optional.add_argument(
        "--keep-originals",
        dest="keep_originals",
        action="store_true",
        default=False,
        help="Don't replace original files; leave new files in output directory.",
    )
    optional.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing.",
    )
    optional.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Output verbose status messages.",
    )

    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
