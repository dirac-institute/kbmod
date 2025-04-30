"""A program to generate stamps from a file of KBMOD results."""

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
    indices : `list[int]` or `None`, optional
        The optional list of result indices to use. If None, all results
        are used.
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
        indices = np.array(indices)
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
    results = Results.read_table(args.input)
    wu = WorkUnit.from_fits(args.workunit, show_progress=args.verbose)

    all_stamps = generate_all_stamps(results, wu, args.radius, args.indices)

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
    )
    parser.add_argument(
        "--results",
        dest="results",
        type=str,
        help="The file path for the input Results file to process.",
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
        "indices",
        nargs="+",
        type=int,
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
    execute(args)
