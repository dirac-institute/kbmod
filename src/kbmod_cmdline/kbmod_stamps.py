"""A program to generate stamps from a file of KBMOD results.

To generate stamps from the result trajectories in 'my_results.ecsv' and image data in the
WorkUnit 'my_wu.fits' you would use:

>>> kbmod-stamps --workunit=my_wu.fits --results=my_results.ecsv --outfile=stamps.npy --type=all

The type specifies which stamp type to generate:
  - "all" generates a stamp for each (result, time step) pair.
     The output is a 4-d numpy array of shape (num results, num times, stamp width, stamp width)
  - "mean" generates a stamp mean coadded stamp for each result.
     The output is a 3-d numpy array of shape (num results, stamp width, stamp width)
  - "median" generates a stamp median coadded stamp for each result.
     The output is a 3-d numpy array of shape (num results, stamp width, stamp width)
  - "sum" generates a stamp sum coadded stamp for each result.
     The output is a 3-d numpy array of shape (num results, stamp width, stamp width)

You can specify a subset of indices (rows in the results file) using the --indices flag.
For example "--indices=1,3,5" will generate stamps from rows 1, 3, and 5.
"""

import argparse
import logging
import numpy as np

from kbmod.core.stamp_utils import (
    coadd_mean,
    coadd_median,
    coadd_sum,
    extract_stamp_stack,
)
from kbmod.results import Results
from kbmod.trajectory_utils import predict_pixel_locations
from kbmod.work_unit import WorkUnit


def generate_all_stamps(results, images, radius=10, indices=None):
    """Generate the stamps and save them to an output file.

    Parameters
    ----------
    results : `Results`
        The results file.
    images : `ImageStack`
        The full set of images to use for the stamps.
    radius : `int`
        The stamp radius.
        Default: 10
    indices : `list[int]`, `np.ndarray` or `None`, optional
        The optional list or array of result indices to use.
        If None, all results are used.
        Default: None

    Returns
    -------
    all_stamps : `np.ndarray`
        The R x T x W x W array of stamps where R is the number of results
        to process (from indices), T is the the number of time steps, and
        W is the stamp radius (2 * radius + 1).
    """
    # Extract and validate the stamp generation parameters.
    if radius <= 0:
        raise ValueError(f"Stamp radius must be > 0 (received {radius}).")
    width = 2 * radius + 1

    if indices is None:
        indices = np.arange(len(results))
    else:
        indices = np.asanyarray(indices)
    num_res = len(indices)

    # Extract the image data we need to build stamps.
    times = np.asarray(images.build_zeroed_times())
    num_times = len(times)
    sci_data = [images.get_single_image(i).get_science_array() for i in range(num_times)]

    # Generate the stamps.
    xvals = predict_pixel_locations(times, results["x"], results["vx"], centered=True, as_int=True)
    yvals = predict_pixel_locations(times, results["y"], results["vy"], centered=True, as_int=True)
    all_stamps = np.zeros((num_res, num_times, width, width), dtype=np.float32)
    for out_i, in_i in enumerate(indices):
        all_stamps[out_i, :, :, :] = extract_stamp_stack(sci_data, xvals[in_i, :], yvals[in_i, :], radius)

    return all_stamps


def coadd_all_stamps(all_stamps, coadd_type):
    """Coadd the stamps in a matrix of all stamps.

    Parameters
    ----------
    all_stamps : `np.ndarray`
        The R x T x W x W array of stamps where R is the number of results
        to process (from indices), T is the the number of time steps, and
        W is the stamp radius (2 * radius + 1).
    coadd_type : `str`
        The type of coadd to use. Must be one of 'mean', 'median', or 'sum'.

    Returns
    -------
    coadds : `np.ndarray`
        The R x W x W array of coadd stamps where R is the number of results
        to process (from indices) and W is the stamp radius (2 * radius + 1).
    """
    num_stamps = all_stamps.shape[0]
    width = all_stamps.shape[2]
    coadds = np.zeros((num_stamps, width, width))

    for idx in range(num_stamps):
        if coadd_type == "mean":
            coadds[idx, :, :] = coadd_mean(all_stamps[idx, :, :, :])
        elif coadd_type == "median":
            coadds[idx, :, :] = coadd_median(all_stamps[idx, :, :, :])
        elif coadd_type == "sum":
            coadds[idx, :, :] = coadd_sum(all_stamps[idx, :, :, :])
        else:
            raise ValueError(f"Unrecognized coadd type {coadd_type}")
    return coadds


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        print("KBMOD Stamp Generation:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
        logging.basicConfig(level=logging.DEBUG)

    # Load the results and the image data.
    results = Results.read_table(args.results)
    wu = WorkUnit.from_fits(args.workunit, show_progress=args.verbose)

    # Parse the indices and validate them.
    if args.indices is None or args.indices == "":
        indices = np.arange(len(results))
    else:
        indices = np.array(args.indices.split(","), dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(results)):
            raise ValueError(f"Invalid indices. Values must be in [0, {len(results)-1}].")

    all_stamps = generate_all_stamps(results, wu.im_stack, args.radius, indices)

    if args.coadd_type == "all":
        np.save(args.outfile, all_stamps)
    else:
        coadds = coadd_all_stamps(all_stamps, args.coadd_type)
        np.save(args.outfile, coadds)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-stamps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to create stamps from KBMOD results data.",
    )
    parser.add_argument(
        "--workunit",
        dest="workunit",
        type=str,
        help="The file path for the workunit file with the image data.",
        required=True,
    )
    parser.add_argument(
        "--results",
        dest="results",
        type=str,
        help="The file path for the input Results file to process.",
        required=True,
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-o",
        "--outfile",
        help="File path to store the resulting stamps.",
        type=str,
        dest="outfile",
        required=False,
        default="./stamps.npy",
    )
    optional.add_argument(
        "-r",
        "--radius",
        type=int,
        default=10,
        dest="radius",
        help="The stamp radius in pixels.",
    )
    optional.add_argument(
        "--type",
        type=str,
        default="all",
        dest="coadd_type",
        help="The stamp type. Must be one of 'all', 'sum', 'mean', or 'median'.",
    )
    optional.add_argument(
        "--indices",
        type=str,
        default="",
        help="A comma separated list of indices to extract. If not provided, uses all results.",
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
