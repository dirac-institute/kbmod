"""A program to compute and display basic statistics for various KBMOD
files such as WorkUnit or Results data.

To generate the stats for the WorkUnit file 'my_input.fits` you would use:

>>> kbmod_stats --workunit=my_input.fits

To generate the stats for the results file 'my_input.ecsv` you would use:

>>> kbmod_stats --results=my_input.ecsv
"""

import argparse
import logging
import numpy as np

from kbmod.results import Results
from kbmod.work_unit import WorkUnit


def stat_result_file(filename, verbose=False):
    """Compute and display the statistics for a given Results file.

    Parameters
    ----------
    filename : `str`
        The name of the input results file.
    verbose : `bool`
        Display verbose debugging output.
    """
    # Load the results and the image data.
    print("-" * len(filename))
    print(f"Stats for Results File '{filename}'")
    print("-" * len(filename))

    results = Results.read_table(filename)
    print(f"\nNumber of results: {len(results)}")
    print("Columns:")
    for col in results.colnames:
        print(f"  {col}")
    if results.mjd_mid is None:
        print("Times: NOT PROVIDED")
    else:
        print(f"Times: {results.mjd_mid}")

    print(f"\nLikelihood Histogram")
    counts, bins = np.histogram(results["likelihood"], bins=10)
    print(" likelihood |  count\n---------------------")
    for edge, count in zip(bins, counts):
        print(f"  {edge:8.2f}  |  {count:5}")

    print("Number of Observations Histogram")
    print("\n num_obs |  count\n------------------")
    max_count = np.max(results["obs_count"]) + 1
    counts, bins = np.histogram(results["obs_count"], bins=np.arange(max_count))
    for edge, count in zip(bins, counts):
        if count > 0:
            print(f"  {edge:5}  |  {count:5}")


def stat_workunit_file(filename, verbose=False):
    """Compute and display the statistics for a given WorkUnit file.

    Parameters
    ----------
    filename : `str`
        The name of the input work unit file.
    verbose : `bool`
        Display verbose debugging output.
    """
    # Load the results and the image data.
    print("-" * len(filename))
    print(f"Stats for WorkUnit File '{filename}'")
    print("-" * len(filename))

    wu = WorkUnit.from_fits(filename, show_progress=verbose)
    wu.print_stats()

    print("\nConstituent Metadata:")
    print(wu.org_img_meta)


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        print("KBMOD Stats:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
        logging.basicConfig(level=logging.DEBUG)

    if args.results is not None and len(args.results) > 0:
        stat_result_file(args.results)
    if args.workunit is not None and len(args.workunit) > 0:
        stat_workunit_file(args.workunit, verbose=args.verbose)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-stats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to compute and display basic statistics for KBMOD files.",
    )
    parser.add_argument(
        "--results",
        dest="results",
        type=str,
        help="The file path for the input Results file to process.",
        required=False,
    )
    parser.add_argument(
        "--workunit",
        dest="workunit",
        type=str,
        help="The file path for the input WorkUnit file to process.",
        required=False,
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
