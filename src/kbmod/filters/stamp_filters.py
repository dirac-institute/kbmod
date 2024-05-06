"""A series of Filter subclasses for processing basic stamp information.

The filters in this file all operate over simple statistics based on the
stamp pixels.
"""

import numpy as np
import time

from kbmod.configuration import SearchConfiguration
from kbmod.results import Results
from kbmod.search import (
    HAS_GPU,
    ImageStack,
    RawImage,
    StampCreator,
    StampParameters,
    StampType,
    Logging,
)


logger = Logging.getLogger(__name__)


def extract_search_parameters_from_config(config):
    """Create an initialized StampParameters object from the configuration settings
    while doing some validity checking.

    Parameters
    ----------
    config : `SearchConfiguration`
        The configuration object.

    Returns
    -------
    params : `StampParameters`
        The StampParameters object with all fields set.

    Raises
    ------
    Raises a ``ValueError`` if parameter validation fails.
    Raises a ``KeyError`` if a required parameter is not found.
    """
    params = StampParameters()

    # Construction parameters
    params.radius = config["stamp_radius"]
    if params.radius < 0:
        raise ValueError(f"Invalid stamp radius {params.radius}")

    stamp_type = config["stamp_type"]
    if stamp_type == "cpp_median" or stamp_type == "median":
        params.stamp_type = StampType.STAMP_MEDIAN
    elif stamp_type == "cpp_mean" or stamp_type == "mean":
        params.stamp_type = StampType.STAMP_MEAN
    elif stamp_type == "cpp_sum" or stamp_type == "sum":
        params.stamp_type = StampType.STAMP_SUM
    else:
        raise ValueError(f"Unrecognized stamp type: {stamp_type}")

    # Filtering parameters (with validity checking)
    params.do_filtering = config["do_stamp_filter"]
    params.center_thresh = config["center_thresh"]

    peak_offset = config["peak_offset"]
    if len(peak_offset) != 2:
        raise ValueError(f"Expected length 2 list for peak_offset. Found {peak_offset}")
    params.peak_offset_x = peak_offset[0]
    params.peak_offset_y = peak_offset[1]

    mom_lims = config["mom_lims"]
    if len(mom_lims) != 5:
        raise ValueError(f"Expected length 5 list for mom_lims. Found {mom_lims}")
    params.m20_limit = mom_lims[0]
    params.m02_limit = mom_lims[1]
    params.m11_limit = mom_lims[2]
    params.m10_limit = mom_lims[3]
    params.m01_limit = mom_lims[4]

    return params


def get_coadds_and_filter_results(result_data, im_stack, stamp_params, chunk_size=1000000):
    """Create the co-added postage stamps and filter them based on their statistical
     properties. Results with stamps that are similar to a Gaussian are kept.

    Parameters
    ----------
    result_data : `Results`
        The current set of results. Modified directly.
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.
    stamp_params : `StampParameters` or `SearchConfiguration`
        The filtering parameters for the stamps.
    chunk_size : `int`
        How many stamps to load and filter at a time. Used to control memory.
    """
    num_results = len(result_data)

    if type(stamp_params) is SearchConfiguration:
        stamp_params = extract_search_parameters_from_config(stamp_params)

    if num_results <= 0:
        logger.debug("Stamp Filtering : skipping, othing to filter.")
    else:
        logger.debug(f"Stamp filtering {num_results} results.")
        logger.debug(f"Using filtering params: {stamp_params}")
        logger.debug(f"Using chunksize = {chunk_size}")

    trj_list = result_data.make_trajectory_list()
    keep_row = [False] * num_results
    stamps_to_keep = []

    # Run the stamp creation and filtering in batches of chunk_size.
    start_time = time.time()
    start_idx = 0
    while start_idx < num_results:
        end_idx = min([start_idx + chunk_size, num_results])
        slice_size = end_idx - start_idx

        # Create a subslice of the results and the Boolean indices.
        # Note that the sum stamp type does not filter out lc_index.
        trj_slice = trj_list[start_idx:end_idx]
        if stamp_params.stamp_type != StampType.STAMP_SUM and "obs_valid" in result_data.colnames:
            bool_slice = result_data["obs_valid"][start_idx:end_idx]
        else:
            # Use all the indices for each trajectory.
            bool_slice = [[True] * im_stack.img_count() for _ in range(slice_size)]

        # Create and filter the results, using the GPU if there is one and enough
        # trajectories to make it worthwhile.
        stamps_slice = StampCreator.get_coadded_stamps(
            im_stack,
            trj_slice,
            bool_slice,
            stamp_params,
            HAS_GPU and len(trj_slice) > 100,
        )
        # TODO: a way to avoid a copy here would be to do
        # np.array([s.image for s in stamps], dtype=np.single, copy=False)
        # but that could cause a problem with reference counting at the m
        # moment. The real fix is to make the stamps return Image not
        # RawImage and avoid reference to an private attribute and risking
        # collecting RawImage but leaving a dangling ref to the attribute.
        # That's a fix for another time so I'm leaving it as a copy here
        for ind, stamp in enumerate(stamps_slice):
            if stamp.width > 1:
                stamps_to_keep.append(np.array(stamp.image))
                keep_row[start_idx + ind] = True

        # Move to the next chunk.
        start_idx += chunk_size

    # Do the actual filtering of results
    result_data.filter_rows(keep_row, label="stamp_filter")

    # Append the coadded stamps to the results. We do this after the filtering
    # so we are not adding a jagged array.
    result_data.table["stamp"] = np.array(stamps_to_keep)

    logger.debug(f"Keeping {len(result_data)} results")
    logger.debug("{:.2f}s elapsed".format(time.time() - start_time))


def append_all_stamps(result_data, im_stack, stamp_radius):
    """Get the stamps for the final results from a kbmod search. These are appended
    onto the corresponding entries in a ResultList.

    Parameters
    ----------
    result_data : `Result` or `ResultList`
        The current set of results. Modified directly.
    im_stack : `ImageStack`
        The stack of images.
    stamp_radius : `int`
        The radius of the stamps to create.
    """
    if type(result_data) is Results:
        all_stamps = []
        for trj in result_data.make_trajectory_list():
            stamps = StampCreator.get_stamps(im_stack, trj, stamp_radius)
            all_stamps.append(np.array([stamp.image for stamp in stamps]))
        result_data.table["all_stamps"] = np.array(all_stamps)
    else:
        # TODO: Remove once we fully replace ResultList with Results
        for row in result_data.results:
            stamps = StampCreator.get_stamps(im_stack, row.trajectory, stamp_radius)
            row.all_stamps = np.array([stamp.image for stamp in stamps])
