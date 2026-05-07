"""A program to ingest LSST Butler collections as KBMOD ImageCollections.

This script allows you to ingest collections from an LSST Butler repository into KBMOD's ImageCollection format.

To use this script, you need to have the LSST stack installed and set up properly and KBMOD available in your Python environment.
You however do not need GPU access within KBMOD to use this script.

To ingest a one or more collections from a Butler repository with a specific dataset type, you can run:
        `python kbmod_build_ic.py <repo_path> <datasetType> --collections <collection1> <collection2> ... --output_dir <output_directory> --overwrite`

Examples of datasetTypes include `calexp`, `preliminary_visit_image`, or `difference_image`.

If you want to ingest all Butler collections matching a regex pattern, you can run:
        `python kbmod_build_ic.py <repo_path> <datasetType> --collection_regex <regex_pattern> --output_dir <output_directory> --overwrite`

The ingested ImageCollections will be saved as `.collection` files in the specified output directory, with the original Butler collection name as the filename (substituting slashes with underscores).

For testing purposes, you can run with the `--dry` flag to see the sizes of the Butler collections without actually ingesting them, or
you can use the `--max_exposures` flag to limit the number of exposures processed per collection.
"""

import argparse
import hashlib
import json
import logging
import os
import re

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    print(f"Importing lsst.daf.butler")
    import lsst.daf.butler as dafButler

    print(f"butler imported")
except ImportError:
    raise ImportError(
        "LSST stack not found. Please install the LSST stack to use this tool. (you may need to re-run `setup lsst_distrib`)"
    )

import kbmod
from kbmod.standardizers import ButlerStandardizerConfig

logger = kbmod.Logging.getLogger(__name__)

# Default Rubin mask flags to use if none are provided
RUBIN_MASK_FLAGS = [
    "BAD",
    "CLIPPED",
    "CR",
    "CROSSTALK",
    "DETECTED_NEGATIVE",
    "EDGE",
    "INEXACT_PSF",
    "INTRP",
    "ITL_DIP",
    "NOT_DEBLENDED",
    "NO_DATA",
    "REJECTED",
    "SAT",
    "SAT_TEMPLATE",
    "SENSOR_EDGE",
    "SPIKE",
    "STREAK",
    "SUSPECT",
    "UNMASKEDNAN",
    "VIGNETTED",
]


def _process_refs_chunk_threaded(refs_chunk, butler_or_repo, config, fail_on_error):
    """
    Worker function for threaded parallel processing.
    Creates its own Butler instance to avoid contention.

    Parameters
    ----------
    refs_chunk : list
        List of Butler DatasetRefs to process.
    butler_or_repo : str or dafButler.Butler
        Path to the Butler repository or an existing Butler instance.
    config : ButlerStandardizerConfig
        Configuration for the ButlerStandardizer.
    fail_on_error : bool
        Whether to fail on standardization errors.

    Returns
    -------
    ImageCollection or None
    """
    try:
        if isinstance(butler_or_repo, str):
            worker_butler = dafButler.Butler(butler_or_repo)
        else:
            worker_butler = butler_or_repo

        ic = kbmod.ImageCollection.fromTargets(
            refs_chunk,
            butler=worker_butler,
            force="ButlerStandardizer",
            config=config,
            fail_on_error=fail_on_error,
        )
        return ic
    except Exception as e:
        logger.error(f"Worker error: {e}")
        return None


def ingest_collection(
    butler,
    repo,
    collection_name,
    datasetType,
    butler_standardizer_config,
    target=None,
    max_exposures=None,
    output_dir=None,
    overwrite=False,
    fail_on_error=False,
    exclude_bands=None,
    num_workers=1,
    refs_pbar=None,
    cone_region=None,
    chunk_size=None,
):
    """
    Ingest a single collection from the LSST Butler repository into a KBMOD ImageCollection, which is then saved as a .collection file.

    Parameters
    ----------
    butler : dafButler.Butler
        The LSST Butler instance to use for querying datasets.
    collection_name : str
        The name of the collection to ingest.
    datasetType : str
        The dataset type to query from the collection (e.g., 'preliminary_visit_image', 'difference_image').
    butler_standardizer_config : ButlerStandardizerConfig
        Configuration for the ButlerStandardizer to use when standardizing images.
    target: str, optional
        The target name to use for the collection. If provided, will be used to filter collections.
    max_exposures : int, optional
        Maximum number of exposures to process from the collection. If None, all exposures are processed.
    output_dir : str, optional
        Directory to write the ImageCollection file. If None, no file is written.
    overwrite : bool, optional
        If True, overwrite existing ImageCollection files. Default is False.
    fail_on_error : bool, optional
        If True, fail the entire ingestion of a collection if any images for the collection failed to standardize. Default is False.
    exclude_bands : list of str, optional
        List of band letters to exclude (e.g., ['u', 'y']). Filters by physical_filter first character.
    num_workers : int, optional
        Number of parallel workers (default: 1 = serial processing).
    refs_pbar : tqdm, optional
        Optional outer progress bar to update with ref counts.
    """
    if output_dir is not None:
        # Generate an output path based on the collection name and replace slashes with underscores
        output_collection_name = collection_name.replace("/", "_")
        output_path = os.path.join(output_dir, f"{output_collection_name}.collection")
        if not overwrite and os.path.exists(output_path):
            logger.info(f"Skipping {collection_name} as it already exists.")
            return
        logger.info(f"Preparing to use output path {output_path} for {collection_name}")

    # Get all butler references for the specified dataset type in this collection
    # Build a where clause and bind dict for the Butler query
    where_clauses = []
    bind = {}
    if target is not None:
        where_clauses.append(f"instrument='LSSTCam' and exposure.target_name='{target}'")
    if cone_region is not None:
        where_clauses.append("visit_detector_region.region OVERLAPS cone_region")
        bind["cone_region"] = cone_region

    where_str = " AND ".join(where_clauses) if where_clauses else None

    try:
        if where_str is not None:
            refs = butler.query_datasets(
                datasetType,
                where=where_str,
                bind=bind if bind else None,
                collections=[collection_name],
                limit=None,
            )
        else:
            refs = butler.registry.queryDatasets(datasetType, collections=[collection_name])
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        return
    refs = list(refs)
    if max_exposures is not None:
        refs = refs[: min(len(refs), max_exposures)]
        logger.info(f"Limiting to first {max_exposures} exposures for collection {collection_name}")
    if not refs:
        logger.debug(f"No datasets found for {datasetType} in {collection_name}.")
        return

    # Filter by band if exclude_bands is specified
    if exclude_bands:
        original_count = len(refs)
        print(f"Excluding bands {exclude_bands} from {original_count} refs")
        filtered_refs = []
        exclude_bands = set(exclude_bands)
        for r in refs:
            if r.dataId.get("physical_filter", "x")[0] not in exclude_bands:
                filtered_refs.append(r)
        filtered_count = original_count - len(filtered_refs)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} refs with excluded bands {exclude_bands}")
        if not filtered_refs:
            logger.debug(f"No datasets remaining after band filtering for {collection_name}.")
            return
        # Update refs to filtered refs
        refs = filtered_refs

    # Sort refs by visit then detector for deterministic chunking and chronological processing
    logger.info(f"Starting to sort {len(refs)} refs")
    refs.sort(key=lambda r: (r.dataId.get("visit", 0), r.dataId.get("detector", 0)))

    total_refs = len(refs)
    logger.info(f"Finished sorting {total_refs} refs")

    # Update outer progress bar description if provided
    if refs_pbar is not None:
        refs_pbar.set_postfix_str(f"{total_refs} refs")

    # Determine chunk size
    if chunk_size is not None:
        cs = chunk_size
    elif num_workers > 1:
        cs = max(1, (total_refs + num_workers - 1) // num_workers)
    else:
        cs = total_refs

    chunks = [refs[i : i + cs] for i in range(0, total_refs, cs)]

    # Setup caching directory if output_dir is provided
    cache_dir = None
    manifest_path = None
    if output_dir is not None:
        output_collection_name = collection_name.replace("/", "_")
        cache_dir = os.path.join(output_dir, "intermediates", output_collection_name)
        os.makedirs(cache_dir, exist_ok=True)
        manifest_path = os.path.join(cache_dir, "manifest.parquet")

    chunks_to_process = []
    ics = []
    new_manifest_records = []

    for idx, chunk in enumerate(chunks):
        # Create a unique hash for this chunk based on its references
        chunk_str = json.dumps(
            [(r.dataId.get("visit", 0), r.dataId.get("detector", 0)) for r in chunk], sort_keys=True
        )
        chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()

        cached_path = None
        if cache_dir is not None:
            chunk_filename = f"chunk_{chunk_hash}.collection"
            cached_path = os.path.join(cache_dir, chunk_filename)
            for r in chunk:
                new_manifest_records.append(
                    {
                        "chunk_filename": chunk_filename,
                        "visit": r.dataId.get("visit"),
                        "detector": r.dataId.get("detector"),
                    }
                )

        if cached_path and os.path.exists(cached_path):
            logger.info(f"Loading cached chunk {idx+1}/{len(chunks)} from {cached_path}")
            try:
                ic = kbmod.ImageCollection.read(cached_path)
                ics.append((idx, ic))
                continue
            except Exception as e:
                logger.warning(f"Failed to load cached chunk {cached_path}, will reprocess: {e}")

        chunks_to_process.append((idx, chunk, cached_path))

    if cache_dir is not None and new_manifest_records:
        df_new = pd.DataFrame(new_manifest_records)
        if os.path.exists(manifest_path):
            try:
                df_old = pd.read_parquet(manifest_path)
                df_combined = pd.concat([df_old, df_new]).drop_duplicates(ignore_index=True)
            except Exception as e:
                logger.warning(f"Failed to read existing manifest {manifest_path}: {e}")
                df_combined = df_new
        else:
            df_combined = df_new
        df_combined.to_parquet(manifest_path, index=False)

    if chunks_to_process:
        if num_workers > 1:
            logger.info(
                f"Processing {len(chunks_to_process)} chunks with {num_workers} processes (each with own Butler)"
            )
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for idx, chunk, cached_path in chunks_to_process:
                    future = executor.submit(
                        _process_refs_chunk_threaded, chunk, repo, butler_standardizer_config, fail_on_error
                    )
                    futures[future] = (idx, cached_path)

                for future in tqdm(as_completed(futures), total=len(futures), desc="Workers", unit="chunk"):
                    idx, cached_path = futures[future]
                    result = future.result()
                    if result is not None:
                        if cached_path:
                            result.write(cached_path, overwrite=True)
                        ics.append((idx, result))
        else:
            logger.info(f"Processing {len(chunks_to_process)} chunks serially")
            for idx, chunk, cached_path in tqdm(chunks_to_process, desc="Chunks", unit="chunk"):
                result = _process_refs_chunk_threaded(
                    chunk, butler, butler_standardizer_config, fail_on_error
                )
                if result is not None:
                    if cached_path:
                        result.write(cached_path, overwrite=True)
                    ics.append((idx, result))

    ics.sort(key=lambda x: x[0])
    ics_list = [ic for idx, ic in ics]

    if not ics_list:
        logger.warning(f"No ImageCollections created for {collection_name}")
        return

    ic = ics_list[0]
    if len(ics_list) > 1:
        ic.vstack(ics_list[1:])
    logger.debug(f"Merged {len(ics_list)} ImageCollections")

    logger.debug(f"Created ImageCollection for collection {collection_name} with {len(ic)} images")
    ic["collection"] = collection_name

    if output_dir is not None:
        if not overwrite and os.path.exists(output_path):
            return
        ic.write(output_path, overwrite=overwrite)

    logger.info(f"Finished ingesting {collection_name}.")


def execute(args):
    # Instantiate the Butler
    print(f"Instantiating Butler for repo {args.repo}")
    butler = dafButler.Butler(args.repo)

    # Determine collections to ingest
    if args.collections:
        collections = args.collections
    else:
        # Get all collections that match the regex
        print(f"Querying collections matching regex {args.collection_regex}")
        all_collections = butler.registry.queryCollections()
        collections = []
        for col in tqdm(all_collections, desc="Matching regex to collections", unit="col"):
            if re.match(args.collection_regex, col):
                collections.append(col)

    if args.dry:
        # If a dry run, print collection sizes
        print(f"\nDry run: Counting refs per collection...")

        # Build cone search region if specified
        cone_region = None
        if args.cone_ra is not None and args.cone_dec is not None:
            from lsst.sphgeom import Circle, UnitVector3d, LonLat, Angle

            center = UnitVector3d(LonLat.fromDegrees(args.cone_ra, args.cone_dec))
            cone_region = Circle(center, Angle.fromDegrees(args.cone_radius))
            print(
                f"Cone search filter active: {args.cone_radius}° radius circle at RA={args.cone_ra}, Dec={args.cone_dec}"
            )

        collection_sizes = []
        all_unique_refs = set()
        for col in tqdm(collections, desc="Counting", unit="col"):
            try:
                # Build where clause and bind dict
                where_clauses = []
                bind = {}
                if args.target is not None:
                    where_clauses.append(f"instrument='LSSTCam' and exposure.target_name='{args.target}'")
                if cone_region is not None:
                    where_clauses.append("visit_detector_region.region OVERLAPS cone_region")
                    bind["cone_region"] = cone_region

                where_str = " AND ".join(where_clauses) if where_clauses else None

                if where_str is not None:
                    refs = list(
                        butler.query_datasets(
                            args.datasetType,
                            where=where_str,
                            bind=bind if bind else None,
                            collections=[col],
                        )
                    )
                else:
                    refs = list(butler.registry.queryDatasets(args.datasetType, collections=[col]))
                # Apply band filtering for accurate count
                if args.exclude_bands:
                    refs = [
                        r for r in refs if r.dataId.get("physical_filter", "x")[0] not in args.exclude_bands
                    ]
                collection_sizes.append((col, len(refs)))
                # Track unique refs by (visit, detector, instrument) tuple
                for r in refs:
                    key = tuple(sorted(r.dataId.mapping.items()))
                    all_unique_refs.add(key)
            except Exception as e:
                logger.warning(f"Error counting {col}: {e}")
                collection_sizes.append((col, 0))

        # Sort by size descending
        collection_sizes.sort(key=lambda x: x[1], reverse=True)

        total = sum(size for _, size in collection_sizes)
        print(f"\nFound {len(collections)} collections")
        print(f"Total refs (after band filtering): {total:,}")
        print(f"Unique refs across all collections: {len(all_unique_refs):,}")
        print(f"\nTop 10 largest collections:")
        for i, (col, size) in enumerate(collection_sizes[:10], 1):
            print(f"  {i:2}. {col}: {size:,} refs")

        return 0

    # Update a default ButlerStandardizer config based on command-line arguments
    std_config = ButlerStandardizerConfig()
    std_config["grow_mask"] = args.grow_mask
    if args.grow_kernel_shape is not None:
        std_config["grow_kernel_shape"] = tuple(args.grow_kernel_shape)
    std_config["mask_flags"] = args.mask_flags if args.mask_flags else RUBIN_MASK_FLAGS
    if args.wcs_fallback_points is not None:
        std_config["wcs_fallback_points"] = args.wcs_fallback_points
    if args.wcs_fallback_sips_degree is not None:
        std_config["wcs_fallback_sips_degree"] = args.wcs_fallback_sips_degree

    # Build cone search region if specified (for non-dry-run path)
    cone_region = None
    if args.cone_ra is not None and args.cone_dec is not None:
        from lsst.sphgeom import Circle, UnitVector3d, LonLat, Angle

        center = UnitVector3d(LonLat.fromDegrees(args.cone_ra, args.cone_dec))
        cone_region = Circle(center, Angle.fromDegrees(args.cone_radius))
        print(
            f"Cone search filter: {args.cone_radius}° radius circle at RA={args.cone_ra}, Dec={args.cone_dec}"
        )

    # Ingest each collection
    pbar = tqdm(collections, desc="Collections", unit="col")
    for collection_name in pbar:
        pbar.set_description(f"Col: {collection_name[:30]}...")
        ingest_collection(
            butler,
            args.repo,
            collection_name,
            args.datasetType,
            std_config,
            args.target,
            args.max_exposures,
            args.output_dir,
            args.overwrite,
            args.fail_on_error,
            args.exclude_bands,
            args.num_workers,
            refs_pbar=pbar,
            cone_region=cone_region,
            chunk_size=args.chunk_size,
        )


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod_build_ic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Ingest LSST Butler collections as KBMOD ImageCollections.",
    )
    # Required positional arguments
    parser.add_argument(
        "repo",
        help="Path to the LSST Butler repository",
    )
    parser.add_argument(
        "datasetType",
        help="Dataset type to ingest (e.g., 'preliminary_visit_image', 'difference_image')",
    )

    # Optional arguments for collection selection and ingestion. At least one of these must be provided.
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Names of Butler collections to ingest separated by spaces. If provided, cannot also provide --collection_regex.",
    )
    parser.add_argument(
        "--collection_regex",
        help="Regex to match collection names",
    )
    parser.add_argument(
        "--target",
        help="Target name to use for the collection. If provided, will be used to filter collections for this target field.",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        help="Directory to write ImageCollections",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ImageCollections",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print collection sizes instead of ingesting",
    )
    parser.add_argument(
        "--max_exposures",
        type=int,
        help="Maximum number of exposures to process per collection",
    )
    parser.add_argument(
        "--exclude_bands",
        nargs="+",
        default=[],
        help="Bands to exclude by first letter of physical_filter (default: u y). Pass --exclude_bands with no args to include all.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing refs (default: 1 = serial). Higher values can speed up large collections.",
    )
    parser.add_argument(
        "--grow_mask",
        action="store_true",
        help="Enable growing the mask to include neighboring pixels",
    )
    parser.add_argument(
        "--grow_kernel_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="Size of the kernel used for growing the mask, as two integers (rows, cols). Example: --grow_kernel_shape 5 5",
    )
    parser.add_argument(
        "--mask_flags",
        nargs="+",
        help="List of mask flags to use when standardizing images. If not provided, a default set of Rubin mask flags will be used.",
    )
    parser.add_argument(
        "--wcs_fallback_points",
        type=int,
        default=None,
        help="Number of random points to sample across the detector when an astropy WCS cannot be constructed from the Rubin SkyWCS metadata.",
    )
    parser.add_argument(
        "--wcs_fallback_sips_degree",
        type=int,
        default=None,
        help="Degree of the SIP distortion to fit when creating a fallback WCS when an astropy WCS cannot be constructed from the Rubin SkyWCS metadata.",
    )
    parser.add_argument(
        "--fail_on_error",
        action="store_true",
        help="Fail the entire ingestion of a collection if any images for the collection failed to standardize",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of exposures per chunk. If not specified, divides evenly among workers.",
    )
    parser.add_argument(
        "--cone_ra",
        type=float,
        default=None,
        help="RA in degrees for cone search center. Used with --cone_dec and --cone_radius.",
    )
    parser.add_argument(
        "--cone_dec",
        type=float,
        default=None,
        help="Dec in degrees for cone search center. Used with --cone_ra and --cone_radius.",
    )
    parser.add_argument(
        "--cone_radius",
        type=float,
        default=2.0,
        help="Radius in degrees for cone search filter (default: 2.0). Only used when --cone_ra and --cone_dec are specified.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate that at least one of collections or collection_regex is provided (but not both)
    if args.collections is None and args.collection_regex is None:
        parser.error("Must specify either --collections or --collection_regex for ingesting.")
    if args.collections is not None and args.collection_regex is not None:
        parser.error("Cannot specify both --collections and --collection_regex. Use one or the other.")
    if (args.cone_ra is None) != (args.cone_dec is None):
        parser.error("Must specify both --cone_ra and --cone_dec together, or neither.")

    execute(args)


if __name__ == "__main__":
    main()
