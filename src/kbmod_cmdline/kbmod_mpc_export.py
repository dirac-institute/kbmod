"""Export KBMOD search results to MPC 80-column observation files.

Scans directories (or specific files) for ``*.search.parquet`` result files
and writes one MPC-formatted ``.txt`` file per trajectory UUID, plus a
manifest table mapping UUIDs back to source files.

Examples
--------
Export all trajectories from a directory::

    kbmod-mpc-export --input /path/to/results --output-dir /path/to/mpc_output

Export specific UUIDs only::

    kbmod-mpc-export --input /path/to/results --output-dir /path/to/mpc_output \\
        --uuids abc123 def456

Export UUIDs from a file (one per line)::

    kbmod-mpc-export --input /path/to/results --output-dir /path/to/mpc_output \\
        --uuid-file my_uuids.txt
"""

import argparse
import logging

from kbmod.mpc_export import export_results_to_mpc_files

logger = logging.getLogger(__name__)


def execute(args):
    """Run the MPC export from parsed arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        print("KBMOD MPC Export:")
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

    manifest = export_results_to_mpc_files(
        directories=args.input,
        output_dir=args.output_dir,
        uuids=uuids,
        glob_pattern=args.glob_pattern,
        observatory=args.observatory,
        chunk_size=args.chunk_size,
    )

    print(f"\nExported {len(manifest)} MPC file(s) to {args.output_dir}")
    if len(manifest) > 0:
        print(f"Manifest: {args.output_dir}/mpc_export_manifest.parquet")
        print(f"Total observations: {sum(manifest['n_obs'])}")


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-mpc-export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Export KBMOD search results to MPC 80-column observation files. "
            "Produces one .txt file per trajectory UUID and a manifest table."
        ),
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=str,
        required=True,
        help=(
            "One or more directories to scan for result files, or direct " "paths to .search.parquet files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where MPC .txt files and the manifest will be written.",
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
        "--observatory",
        type=str,
        default="X05",
        help="Three character MPC observatory code.",
    )
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
