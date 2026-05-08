"""A program to strip columns from KBMOD results based on various filtering criteria,
including removing specific columns, removing all image data, and removing all
time series data.

To remove all image and time series data, use the 'kbmod-strip-results' command:

>>> kbmod-strip-results --input=fake_results.ecsv --outfile=test.ecsv

By default this removes all image, time series, and columns with only None or NaN values. To
retain any of these types of data, use the appropriate flag:

>>> kbmod-strip-results --input=fake_results.ecsv --outfile=test.ecsv --keep_images
>>> kbmod-strip-results --input=fake_results.ecsv --outfile=test.ecsv --keep_time_series
>>> kbmod-strip-results --input=fake_results.ecsv --outfile=test.ecsv --keep_none

You can also specify specific columns to keep or drop:

>>> kbmod-strip-results --input=fake_results.ecsv --outfile=test.ecsv --keep_columns col1 col2
>>> kbmod-strip-results --input=fake_results.ecsv --outfile=test.ecsv --drop_columns col3 col4

"""

import argparse
import logging
import numpy as np

from kbmod.results import Results

logger = logging.getLogger(__name__)


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        print("KBMOD Results Stripping:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
        logging.basicConfig(level=logging.DEBUG)

    # Parse the keep and drop columns.
    given_keep = set(args.keep_columns) if args.keep_columns is not None else set()
    given_drop = set(args.drop_columns) if args.drop_columns is not None else set()
    if len(given_keep.intersection(given_drop)) > 0:
        raise ValueError(f"Columns cannot be both kept and dropped: {given_keep & given_drop}")

    # Load the results.
    results = Results.read_table(args.input)
    num_res = len(results)
    logger.debug(f"Loaded {num_res} results with columns: {results.colnames}")

    all_drop = set()
    for col in results.colnames:
        # Handle the cases where we always keep or always drop.
        if col in given_keep:
            logger.debug(f"Retaining column {col} because it was specified.")
            continue
        elif col in given_drop:
            logger.debug(f"Dropping column {col} because it was specified.")
            all_drop.add(col)
            continue

        # Try to infer the type of data in the column from it's shape or name.
        num_dims = len(results[col].shape)
        if not args.keep_images:
            if num_dims == 3:
                logger.debug(f"Dropping column {col} because it has 3 dimensions (image stamps).")
                all_drop.add(col)
            elif col.startswith("coadd_"):
                logger.debug(f"Dropping column {col} because it is a coadd image.")
                all_drop.add(col)
            elif col.startswith("stamp"):
                logger.debug(f"Dropping column {col} because it is an image stamp.")
                all_drop.add(col)
        if not args.keep_time_series:
            if num_dims == 2:
                logger.debug(f"Dropping column {col} because it has 2 dimensions (time series).")
                all_drop.add(col)
            elif "_curve" in col:
                logger.debug(f"Dropping column {col} because it is a time series (_curve in name).")
                all_drop.add(col)
        if not args.keep_none:
            entry_is_none = [x is None or (isinstance(x, float) and np.isnan(x)) for x in results[col]]
            if np.all(entry_is_none):
                logger.debug(f"Dropping column {col} because all entries are None or NaN.")
                all_drop.add(col)

        if col not in all_drop:
            logger.debug(f"Retaining column {col}.")

    logger.debug(f"Dropping {len(all_drop)} columns: {all_drop}")
    results.table.remove_columns(list(all_drop))
    logger.debug(f"After stripping, {len(results.colnames)} columns remain: {results.colnames}")

    # Save the modified results file.
    results.write_table(args.outfile)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-strip-results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to strip columns from KBMOD results.",
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

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--keep_images",
        default=False,
        dest="keep_images",
        action="store_true",
        help="Keep image stamps in the output results.",
    )
    optional.add_argument(
        "--keep_time_series",
        default=False,
        dest="keep_time_series",
        action="store_true",
        help="Keep time series data in the output results.",
    )
    optional.add_argument(
        "--keep_none",
        default=False,
        dest="keep_none",
        action="store_true",
        help="Keep columns containing only None or NaN values.",
    )
    optional.add_argument(
        "--keep_columns",
        dest="keep_columns",
        nargs="*",
        help=(
            "The columns to retain in the output Results files (even if they "
            "are removed by another option)."
        ),
    )
    optional.add_argument(
        "--drop_columns",
        dest="drop_columns",
        nargs="*",
        help="The columns to drop from the output Results files.",
    )
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
