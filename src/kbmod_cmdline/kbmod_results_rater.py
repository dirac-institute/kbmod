"""A program to perform post-KBMOD results visualization and classification.

>>> kbmod-rater --input=fake_results.ecsv --outfile=test.ecsv
"""

import argparse
import time

from kbmod.analysis.results_rater import ResultsRater
from kbmod.results import Results


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    # Load the results and the image data.
    results = Results.read_table(args.input)

    rater = ResultsRater(results, outfile=args.outfile)

    # Use a very primitive event loop to keep the program running.
    while rater.is_running:
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-rater",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to visualize and classify.",
    )
    parser.add_argument(
        "--input",
        dest="input",
        type=str,
        help="The file path for the input Results file to process.",
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

    # Run the actual program.
    args = parser.parse_args()
    execute(args)
