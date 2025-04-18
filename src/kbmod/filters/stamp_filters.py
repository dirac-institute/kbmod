"""A series of Filter subclasses for processing basic stamp information.

The filters in this file all operate over simple statistics based on the
stamp pixels.
"""

import numpy as np
import time

from kbmod.configuration import SearchConfiguration
from kbmod.core.stamp_utils import (
    coadd_mean,
    coadd_median,
    coadd_sum,
    coadd_weighted,
    extract_stamp_stack,
)
from kbmod.results import Results
from kbmod.trajectory_utils import predict_pixel_locations
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


def append_coadds(result_data, im_stack, coadd_types, radius, valid_only=True):
    """Append one or more stamp coadds to the results data without filtering.

    result_data : `Results`
        The current set of results. Modified directly.
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.
    coadd_types : `list`
        A list of coadd types to generate. Can be "sum", "mean", and "median".
    radius : `int`
        The stamp radius to use.
    valid_only : `bool`
        Only use stamps from the timesteps marked valid for each trajectory.
    """
    if radius <= 0:
        raise ValueError(f"Invalid stamp radius {radius}")
    width = 2 * radius + 1

    # We can't use valid only if there is not obs_valid column in the data.
    valid_only = valid_only and "obs_valid" in result_data.colnames

    stamp_timer = DebugTimer("computing extra coadds", logger)

    # Copy the image data that we need. The data only copies the references to the numpy arrays.
    num_times = im_stack.img_count()
    sci_data = [im_stack.get_single_image(i).get_science_array() for i in range(num_times)]
    var_data = [im_stack.get_single_image(i).get_variance_array() for i in range(num_times)]
    times = np.asarray(im_stack.build_zeroed_times())

    # Predict the x and y locations in a giant batch.
    num_res = len(result_data)
    xvals = predict_pixel_locations(times, result_data["x"], result_data["vx"], centered=True, as_int=True)
    yvals = predict_pixel_locations(times, result_data["y"], result_data["vy"], centered=True, as_int=True)

    # Allocate space for the coadds in the results table.
    for coadd_type in coadd_types:
        result_data.table[f"coadd_{coadd_type}"] = np.zeros((num_res, width, width))

    # Loop through each trajectory generating the coadds.  We extract the stamp stack once
    # for each trajectory and compute all the coadds from that stack.
    for idx in range(num_res):
        to_include = None if not valid_only else result_data["obs_valid"][idx]
        sci_stack = extract_stamp_stack(sci_data, xvals[idx, :], yvals[idx, :], radius, to_include=to_include)
        sci_stack = np.asanyarray(sci_stack)

        if "mean" in coadd_types:
            result_data[f"coadd_mean"][idx][:, :] = coadd_mean(sci_stack)
        if "median" in coadd_types:
            result_data[f"coadd_median"][idx][:, :] = coadd_median(sci_stack)
        if "sum" in coadd_types:
            result_data[f"coadd_sum"][idx][:, :] = coadd_sum(sci_stack)
        if "weighted" in coadd_types:
            var_stack = extract_stamp_stack(
                var_data, xvals[idx, :], yvals[idx, :], radius, to_include=to_include
            )
            var_stack = np.asanyarray(var_stack)
            result_data[f"coadd_weighted"][idx][:, :] = coadd_weighted(sci_stack, var_stack)

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

    if stamp_radius < 1:
        raise ValueError(f"Invalid stamp radius: {stamp_radius}")
    width = 2 * stamp_radius + 1

    # Copy the image data that we need. The data only copies the references to the numpy arrays.
    num_times = im_stack.img_count()
    sci_data = [im_stack.get_single_image(i).get_science_array() for i in range(num_times)]
    times = np.asarray(im_stack.build_zeroed_times())

    # Predict the x and y locations in a giant batch.
    num_res = len(result_data)
    xvals = predict_pixel_locations(times, result_data["x"], result_data["vx"], centered=True, as_int=True)
    yvals = predict_pixel_locations(times, result_data["y"], result_data["vy"], centered=True, as_int=True)

    all_stamps = np.zeros((num_res, num_times, width, width))
    for idx in range(num_res):
        all_stamps[idx, :, :, :] = extract_stamp_stack(sci_data, xvals[idx, :], yvals[idx, :], stamp_radius)

    # columns between tables.
    result_data.table["all_stamps"] = all_stamps
    stamp_timer.stop()


def _normalize_stamps(stamps, stamp_dimm):
    """Normalize a list of stamps. Used for `filter_stamps_by_cnn`."""
    normed_stamps = []
    sigma_g_coeff = 0.7413
    for stamp in stamps:
        stamp = np.copy(stamp)
        stamp[np.isnan(stamp)] = 0

        per25, per50, per75 = np.percentile(stamp, [25, 50, 75])
        sigmaG = sigma_g_coeff * (per75 - per25)
        stamp[stamp < (per50 - 2 * sigmaG)] = per50 - 2 * sigmaG

        stamp -= np.min(stamp)
        stamp /= np.sum(stamp)
        stamp[np.isnan(stamp)] = 0
        normed_stamps.append(stamp.reshape(stamp_dimm, stamp_dimm))
    return np.array(normed_stamps)


def filter_stamps_by_cnn(result_data, model_path, coadd_type="mean", stamp_radius=10, verbose=False):
    """Given a set of results data, run the the requested coadded stamps through a
    provided convolutional neural network and assign a new column that contains the
    stamp classification, i.e. whether or not the result passed the CNN filter.

    Parameters
    ----------
    result_data : `Result`
        The current set of results. Modified directly.
    model_path : `str`
        Path to the the tensorflow model and weights file.
    coadd_type : `str`
        Which coadd type to use in the filtering. Depends on how the model was trained.
        Default is 'mean', will grab stamps from the 'coadd_mean' column.
    stamp_radius : `int`
        The radius used to generate the stamps. The dimension of the stamps should be
        (stamp_radius * 2) + 1.
    verbose : `bool`
        Verbosity option for the CNN predicition. Off by default.
    """
    from tensorflow.keras.models import load_model

    coadd_column = f"coadd_{coadd_type}"
    if coadd_column not in result_data.colnames:
        raise ValueError("result_data does not have provided coadd type as a column.")

    cnn = load_model(model_path)

    stamps = result_data.table[coadd_column].data
    stamp_dimm = (stamp_radius * 2) + 1
    normalized_stamps = _normalize_stamps(stamps, stamp_dimm)

    # resize to match the tensorflow input
    # will probably not be needed when we switch to PyTorch
    resized_stamps = normalized_stamps.reshape(-1, stamp_dimm, stamp_dimm, 1)

    predictions = cnn.predict(resized_stamps, verbose=verbose)

    classifications = []
    for p in predictions:
        classifications.append(np.argmax(p))

    bool_arr = np.array(classifications) != 0
    result_data.table["cnn_class"] = bool_arr
