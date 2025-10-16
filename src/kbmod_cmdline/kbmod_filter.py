"""A program to apply post-search filtering to KBMOD results.

This script applies a combination of sigma-G, minimum number of observations, minimum likleihood,
and clustering-based filtering. Users can run any of all of them.

To run sigma-g filtering use the 'sigma_g_bnds' command line argument.

>>> kbmod-filter --input=fake_results.ecsv --workunit=test_wu.fits --outfile=test.ecsv --sigma_g_bnds=20,80

Similarly number of observations can be enabled with the 'num_obs' argument and likelihood filtering
with the 'lh_level' argument.

>>> kbmod-filter --input=fake_results.ecsv --workunit=test_wu.fits --outfile=test.ecsv \
    --num_obs=5 --lh_level=10.0

Clustering requires two arguments 'cluster_type' and 'cluster_eps':

>>> kbmod-filter --input=fake_results.ecsv --workunit=test_wu.fits --outfile=test.ecsv \
    --cluster_type=nn_start_end --cluster_eps=100.0
"""

import argparse
import logging
import numpy as np

from kbmod.filters.clustering_filters import apply_clustering
from kbmod.filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from kbmod.results import Results
from kbmod.search import StackSearch
from kbmod.work_unit import WorkUnit


logger = logging.getLogger(__name__)


def sigma_g_filter_results(results, bnds, clip_negative=False, workunit=None):
    """Apply sigma-G filtering to the results table.

    Parameters
    ----------
    results : `Results`
        The input results table.  This is filtered in place.
    bnds : `tuple(float, float)`
        The percentiles for sigmaG filtering, such as (25, 75).
    clip_negative : `bool`
        Remove all negative values prior to computing the percentiles.
        Default: False
    workunit : `WorkUnit`, optional
        The workunit to use if the results data does not contain psi and phi information.
    """
    if len(bnds) != 2 or bnds[0] >= bnds[1] or bnds[0] < 0.0 or bnds[1] > 100.0:
        raise ValueError(f"Invalid sigma-g limits: {sigma_g_lims}")

    if "psi_curve" not in results.colnames or "phi_curve" not in results.colnames:
        if workunit is None:
            raise ValueError("Missing psi-phi data.")

        result_trjs = results.make_trajectory_list()
        logger.debug(f"Generating psi and phi data for {len(result_trjs)} results.")

        search = StackSearch(workunit.im_stack)
        num_times = search.get_num_images()
        psi_phi = search.get_all_psi_phi_curves(result_trjs)
        results.add_psi_phi_data(psi_phi[:, :num_times], psi_phi[:, num_times:])
    else:
        logger.debug(f"Found psi and phi data in results.")

    logger.debug(f"Running sigma-g filtering with bounds {bnds}")
    clipper = SigmaGClipping(bnds[0], bnds[1], 2, clip_negative)
    apply_clipped_sigma_g(clipper, results)


def execute(args):
    """Run the program from the given arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    if args.verbose:
        print("KBMOD Filtering:")
        for key, val in vars(args).items():
            print(f"  {key}: {val}")
        logging.basicConfig(level=logging.DEBUG)

    # Load the results and the image data.
    results = Results.read_table(args.input)
    logger.debug(f"Loaded {len(results)} results to filter.")
    if args.workunit is not None:
        wu = WorkUnit.from_fits(args.workunit, show_progress=args.verbose)
    else:
        wu = None

    # Do sigma-g filtering if bounds are provided.
    if args.sigma_g_bnds is not None:
        split_str = args.sigma_g_bnds.split(",")
        if len(split_str) != 2:
            raise ValueError(f"Invalid sigma-G bounds: {args.sigma_g_bnds}")
        bnds = (float(split_str[0]), float(split_str[1]))

        sigma_g_filter_results(results, bnds, args.clip_negative, workunit=wu)

    # Do num_obs and lh filtering.
    results.filter_rows(results["obs_count"] >= args.num_obs, "num_obs")
    results.filter_rows(results["likelihood"] >= args.lh_level, "lh_level")
    logger.debug(f"After sigma-G filtering, result size = {len(results)}")

    # Perform clustering if a type is provided.
    if args.cluster_type is not None:
        if results.mjd_mid is not None:
            mjd = results.mjd_mid
        elif wu is not None:
            mjd = np.array([wu.im_stack.get_obstime(t) for t in range(wu.im_stack.img_count())])
        else:
            raise ValueError("Time stamps not present in results or workunit.")

        cluster_params = {
            "cluster_type": args.cluster_type,
            "cluster_eps": args.cluster_eps,
            "cluster_v_scale": 1.0,
            "times": np.asarray(mjd),
        }
        logger.debug(f"Clustering results with:\n{cluster_params}")
        apply_clustering(results, cluster_params)
        logger.debug(f"After clustering, result size = {len(results)}")

    # Save the modified results file.
    if len(results) > 0 or args.output_empty_table:
        results.write_table(args.outfile)


def main():
    parser = argparse.ArgumentParser(
        prog="kbmod-filter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A program to apply post-search filtering to KBMOD results.",
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
        "--workunit",
        dest="workunit",
        type=str,
        help=(
            "The file path for the workunit file with the image data. Required "
            "if the input Results table does not have psi and phi information."
        ),
        required=False,
    )
    optional.add_argument(
        "--num_obs",
        dest="num_obs",
        help="The minimum number of observations.",
        default=1,
        type=int,
        required=False,
    )
    optional.add_argument(
        "--lh_level",
        dest="lh_level",
        help="The minimum likelihood value.",
        default=0.0,
        type=float,
        required=False,
    )
    optional.add_argument(
        "--sigma_g_bnds",
        dest="sigma_g_bnds",
        type=str,
        help="The comma separated values for the sigma-g percentiles (e.g. 25,75).",
        default=None,
        required=False,
    )
    optional.add_argument(
        "--clip_negative",
        default=False,
        dest="clip_negative",
        action="store_true",
        help="Clip negative values in sigma-g filtering.",
    )
    optional.add_argument(
        "--cluster_type",
        default=None,
        dest="cluster_type",
        help="The type of clustering to perform.",
    )
    optional.add_argument(
        "--cluster_eps",
        default=1.0,
        type=float,
        dest="cluster_eps",
        help="The clustering threshold in pixels.",
    )
    optional.add_argument(
        "-v",
        "--verbose",
        default=False,
        dest="verbose",
        action="store_true",
        help="Output verbose status messages.",
    )
    optional.add_argument(
        "--output_empty_table",
        default=False,
        dest="output_empty_table",
        action="store_true",
        help="Output an empty table if no results remain after filtering.",
    )

    # Run the actual program.
    args = parser.parse_args()
    execute(args)
