"""A program to merge together multple results files with basic duplicate removal.

>>> kbmod-merge-results --input=file1.ecsv file2.ecsv --outfile=merged.ecsv --dup_thresh=10
"""

import argparse
import numpy as np
import warnings

from kbmod.filters.clustering_filters import NNSweepFilter
from kbmod.results import Results


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if len(args.input) < 1:
        raise ValueError("At least one input file must be specified.")
    results = Results.read_table(args.input[0])

    if args.verbose:
        print("--- Merging results files ---")
        print(f"Read {len(results)} entries from file: {args.input[0]}")

    # Set up the duplicate filters if needed.
    if args.dup_thresh > 0:
        filter_first = NNSweepFilter(cluster_eps=args.dup_thresh, pred_times=[0.0])
        if results.mjd_mid is None:
            warnings.warn("No times found in results, so using 1 day interval.")
            filter_last = NNSweepFilter(cluster_eps=args.dup_thresh, pred_times=[1.0])
        else:
            last_time = results.mjd_mid[-1] - results.mjd_mid[0]
            filter_last = NNSweepFilter(cluster_eps=args.dup_thresh, pred_times=[last_time])
    else:
        filter_first = None
        filter_last = None

    # Add each additional results file, removing duplicates as we go.
    for input_file in args.input[1:]:
        new_results = Results.read_table(input_file)
        if args.verbose:
            print(f"Read {len(new_results)} entries from file: {input_file}")
        results.extend(new_results)

        if filter_first is not None and filter_last is not None:
            keep_inds_first = filter_first.keep_indices(results)
            keep_inds_last = filter_last.keep_indices(results)
            keep_inds = np.union1d(keep_inds_first, keep_inds_last)

            if args.verbose:
                num_removed = len(results) - len(keep_inds)
                print(f"Removed {num_removed} duplicates with threshold={args.dup_thresh} pixels.")

            # Remove duplicates using a nearest-neighbor sweep filter.
            results.filter_rows(keep_inds, "deduplicate")

    # Write out the merged results.
    if args.verbose:
        print(f"Writing {len(results)} total entries to file: {args.outfile}")
    results.write_table(args.outfile)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-merge-results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to merge multiple results files.",
    )
    parser.add_argument(
        "--input",
        dest="input",
        nargs="+",
        help="The file paths for the input Results files to process.",
        required=True,
    )
    parser.add_argument(
        "--outfile",
        help=(
            "File path for the output Results file. Can be the same as the input, "
            "in which case the input is overwritten."
        ),
        type=str,
        dest="outfile",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--dup_thresh",
        type=int,
        default=0,
        dest="dup_thresh",
        help="The maximum distance (in pixels) for two results to be considered duplicates.",
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-v",
        "--verbose",
        default=False,
        dest="verbose",
        action="store_true",
        help="Output verbose status messages.",
    )

    # Run the actual program.
    args = parser.parse_args()
    execute(args)
