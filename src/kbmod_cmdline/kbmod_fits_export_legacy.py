"""Export KBMOD search results to combined FITS and CSV files.

Scans directories (or specific files) for ``*.search.parquet`` result files
and writes a single combined FITS file with DETECTIONS, EXPOSURES, and
TRACK_ID_MAP extensions compatible with the BulkFit orbit fitting tool,
plus an optional CSV file.

The FITS output includes:
- DETECTIONS: Per-detection data (TRACK_ID, EXPNUM, RA, DEC, SIGMA)
- EXPOSURES: Per-exposure data (EXPNUM, RA, DEC, MJD_MID)
- TRACK_ID_MAP: Mapping from TRACK_ID to UUID and source file

The primary HDU header contains reference frame values (RA0, DEC0, MJD0,
X0, Y0, Z0) derived from the ImageCollection associated with the results.

Examples
--------
Export all trajectories from a directory::

    kbmod-fits-export --input /path/to/results --output /path/to/combined.fits

Export with CSV output as well::

    kbmod-fits-export --input /path/to/results --output /path/to/combined.fits \\
        --csv /path/to/combined.csv

Export specific UUIDs only::

    kbmod-fits-export --input /path/to/results --output /path/to/combined.fits \\
        --uuids abc123 def456

Export UUIDs from a file (one per line)::

    kbmod-fits-export --input /path/to/results --output /path/to/combined.fits \\
        --uuid-file my_uuids.txt

Use a custom astrometric uncertainty::

    kbmod-fits-export --input /path/to/results --output /path/to/combined.fits \\
        --sigma-default 0.5

Prepend existing objects from MPC 80-column files (for orbit linking)::

    kbmod-fits-export --input /path/to/results --output /path/to/combined.fits \\
        --existing-objects-dir /path/to/mpc_files
"""

import argparse
import logging

from kbmod.fits_export import export_results

logger = logging.getLogger(__name__)


def execute(args):
    """Run the FITS export from parsed arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        print("KBMOD FITS Export:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
    else:
        logging.basicConfig(level=logging.INFO)

    # Collect UUIDs from --uuids and/or --uuid-file
    uuids = None
    uuid_list = []

    if args.uuids:
        uuid_list.extend(args.uuids)

    if args.uuid_file:
        with open(args.uuid_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    uuid_list.append(line)

    if uuid_list:
        uuids = uuid_list
        logger.info(f"Filtering to {len(uuids)} specific UUID(s)")

    detections_table, exposures_table, trackid_map_table = export_results(
        directories=args.input,
        output_path=args.output,
        output_csv=args.csv,
        uuids=uuids,
        glob_pattern=args.glob_pattern,
        chunk_size=args.chunk_size,
        sigma_default=args.sigma_default,
        existing_objects_dir=args.existing_objects_dir,
    )

    print(f"\nExported to {args.output}")
    print(f"  Detections: {len(detections_table)}")
    print(f"  Exposures: {len(exposures_table)}")
    print(f"  Track IDs: {len(trackid_map_table)}")
    if args.csv:
        print(f"  CSV file: {args.csv}")


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-fits-export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Export KBMOD search results to a combined FITS file compatible with "
            "BulkFit orbit fitting. Produces a FITS file with DETECTIONS, EXPOSURES, "
            "and TRACK_ID_MAP extensions, plus an optional CSV file."
        ),
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=str,
        required=True,
        help=(
            "One or more directories to scan for result files, or direct "
            "paths to .search.parquet files."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output FITS file.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path for CSV output (detections table).",
    )

    optional = parser.add_argument_group("Filtering")
    optional.add_argument(
        "--uuids",
        nargs="+",
        type=str,
        default=None,
        help="Export only these UUIDs (space-separated).",
    )
    optional.add_argument(
        "--uuid-file",
        type=str,
        default=None,
        help="Path to a file containing UUIDs to export (one per line).",
    )

    config = parser.add_argument_group("Configuration")
    config.add_argument(
        "--glob-pattern",
        type=str,
        default="*.search.parquet",
        help="Glob pattern for finding result files in directories.",
    )
    config.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows to process at a time (for memory efficiency).",
    )
    config.add_argument(
        "--sigma-default",
        type=float,
        default=1.0,
        help="Default astrometric uncertainty in arcsec when not available.",
    )
    config.add_argument(
        "--existing-objects-dir",
        type=str,
        default=None,
        help=(
            "Directory containing MPC 80-column format files (*.txt) for existing "
            "objects. When provided, enables prepend mode where each KBMOD result "
            "is duplicated with existing object observations prepended."
        ),
    )
    config.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug output.",
    )

    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
