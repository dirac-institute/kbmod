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
    DebugTimer,
    ImageStack,
    RawImage,
    StampParameters,
    StampType,
    Logging,
    get_stamps,
    get_coadded_stamps,
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
    elif stamp_type == "weighted":
        params.stamp_type = StampType.STAMP_VAR_WEIGHTED
    else:
        raise ValueError(f"Unrecognized stamp type: {stamp_type}")

    return params


def make_coadds(result_data, im_stack, stamp_params, chunk_size=1_000_000, colname="stamp"):
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
        Default: 100_000
    colname : `str`
        The column in which to save the coadded stamp.
        Default: "stamp"
    """
    num_results = len(result_data)
    if num_results <= 0:
        logger.info("Creating coadds : skipping, nothing to filter.")

        # We still add the (empty) column so we keep different table's
        # columns consistent.
        result_data.table["stamp"] = np.array([])
        return

    if type(stamp_params) is SearchConfiguration:
        stamp_params = extract_search_parameters_from_config(stamp_params)

    stamp_timer = DebugTimer(f"creating coadd stamps", logger)
    logger.info(f"Creating coadds of {num_results} results in column={colname}.")
    logger.debug(f"Using filtering params: {stamp_params}")
    logger.debug(f"Using chunksize = {chunk_size}")

    trj_list = result_data.make_trajectory_list()
    keep_row = [False] * num_results
    stamps_to_keep = []

    # Run the stamp creation and filtering in batches of chunk_size.
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
        stamps_slice = get_coadded_stamps(
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
            stamps_to_keep.append(np.array(stamp.image))
            keep_row[start_idx + ind] = True

        # Move to the next chunk.
        start_idx += chunk_size

    # Append the coadded stamps to the results. We do this after the filtering
    # so we are not adding a jagged array.
    result_data.table[colname] = np.array(stamps_to_keep)
    stamp_timer.stop()


def append_coadds(result_data, im_stack, coadd_types, radius, chunk_size=100_000):
    """Append one or more stamp coadds to the results data without filtering.

    result_data : `Results`
        The current set of results. Modified directly.
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.
    coadd_types : `list`
        A list of coadd types to generate. Can be "sum", "mean", and "median".
    radius : `int`
        The stamp radius to use.
    chunk_size : `int`
        How many stamps to load and filter at a time. Used to control memory.
        Default: 100_000
    """
    if radius <= 0:
        raise ValueError(f"Invalid stamp radius {radius}")
    stamp_timer = DebugTimer("computing extra coadds", logger)

    params = StampParameters()
    params.radius = radius

    # Loop through all the coadd types in the list, generating a corresponding stamp.
    for coadd_type in coadd_types:
        logger.info(f"Adding coadd={coadd_type} for all results.")

        if coadd_type == "median":
            params.stamp_type = StampType.STAMP_MEDIAN
        elif coadd_type == "mean":
            params.stamp_type = StampType.STAMP_MEAN
        elif coadd_type == "sum":
            params.stamp_type = StampType.STAMP_SUM
        elif coadd_type == "weighted":
            params.stamp_type = StampType.STAMP_VAR_WEIGHTED
        else:
            raise ValueError(f"Unrecognized stamp type: {coadd_type}")

        # Do the generation (without filtering).
        make_coadds(
            result_data,
            im_stack,
            params,
            chunk_size=chunk_size,
            colname=f"coadd_{coadd_type}",
        )
    stamp_timer.stop()


def append_all_stamps(result_data, im_stack, stamp_radius):
    """Get the stamps for the final results from a kbmod search. These are appended
    onto the corresponding entries in a ResultList.

    Parameters
    ----------
    result_data : `Result`
        The current set of results. Modified directly.
    im_stack : `ImageStack`
        The stack of images.
    stamp_radius : `int`
        The radius of the stamps to create.
    """
    logger.info(f"Appending all stamps for {len(result_data)} results")
    stamp_timer = DebugTimer("computing all stamps", logger)

    all_stamps = []
    for trj in result_data.make_trajectory_list():
        stamps = get_stamps(im_stack, trj, stamp_radius)
        all_stamps.append(np.array([stamp.image for stamp in stamps]))

    # We add the column even if it is empty so we can have consistent
    # columns between tables.
    result_data.table["all_stamps"] = np.array(all_stamps)
    stamp_timer.stop()
