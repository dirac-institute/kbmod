"""A program to compute and display basic statistics for results data.

To generate the stats for the results file 'my_input.ecsv` you would use:

>>> kbmod_stats --input=my_input.ecsv
"""

import argparse
import numpy as np

from kbmod.results import Results


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

    # Load the results and the image data.
    print("-" * 60)
    print(f"Stats for Results File '{args.input}'")
    print("-" * 60)

    results = Results.read_table(args.input)
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


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-stats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to compute and display basic statistics for results data.",
    )
    parser.add_argument(
        "--input",
        dest="input",
        type=str,
        help="The file path for the input Results file to process.",
        required=True,
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
