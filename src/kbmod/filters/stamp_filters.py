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
    Logging,
)


logger = Logging.getLogger(__name__)


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
