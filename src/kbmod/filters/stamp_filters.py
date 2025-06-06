"""A series of Filter subclasses for processing basic stamp information.

The filters in this file all operate over simple statistics based on the
stamp pixels.
"""

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.core.stamp_utils import (
    coadd_mean,
    coadd_median,
    coadd_sum,
    coadd_weighted,
    extract_stamp_stack,
)
from kbmod.search import DebugTimer, Logging
from kbmod.trajectory_utils import predict_pixel_locations
from kbmod.util_functions import mjd_to_day


logger = Logging.getLogger(__name__)

MODEL_TYPES = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}

# Mock up of the model to ensure everything loads correctly.


def modify_resnet_input_channels(model, num_channels):
    # Get the first convolutional layer
    first_conv_layer = model.conv1

    # Create a new convolutional layer with the desired number of input channels
    new_conv_layer = nn.Conv2d(
        in_channels=num_channels,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias,
    )

    # Replace the first convolutional layer in the model
    model.conv1 = new_conv_layer

    return model


class _KBMLModel(nn.Module):
    def __init__(self, model, weights, shape):
        super().__init__()

        self.model = model

        # Modify the input channels to 1 (e.g., for grayscale images)
        # print(shape[0])
        self.model = modify_resnet_input_channels(model=model, num_channels=shape[0])

    def forward(self, x):
        # if labels are passed to forward as part
        # of the infer step of training, just pass along the stamps.
        if isinstance(x, tuple):
            x, _ = x
        return self.model(x)


def append_coadds(result_data, im_stack, coadd_types, radius, valid_only=True, nightly=False):
    """Append one or more stamp coadds to the results data without filtering.

    result_data : `Results`
        The current set of results. Modified directly.
    im_stack : `ImageStackPy`
        The images from which to build the co-added stamps.
    coadd_types : `list`
        A list of coadd types to generate. Can be "sum", "mean", and "median".
    radius : `int`
        The stamp radius to use.
    valid_only : `bool`
        Only use stamps from the timesteps marked valid for each trajectory.
    nightly : `bool`
        Break up the stamps to a single coadd per-calendar day.
    """
    if radius <= 0:
        raise ValueError(f"Invalid stamp radius {radius}")
    width = 2 * radius + 1

    # We can't use valid only if there is not obs_valid column in the data.
    valid_only = valid_only and "obs_valid" in result_data.colnames

    stamp_timer = DebugTimer("computing extra coadds", logger)

    # Access the time data we need. If we are doing nightly, compute the
    # strings for each time. Use "" to mean all times.
    times = im_stack.zeroed_times
    day_strs = np.array([f"_{mjd_to_day(t)}" for t in im_stack.times])
    if nightly:
        days_to_use = np.unique(day_strs)
    else:
        days_to_use = []

    # Predict the x and y locations in a giant batch.
    num_res = len(result_data)
    xvals = predict_pixel_locations(times, result_data["x"], result_data["vx"], centered=True, as_int=True)
    yvals = predict_pixel_locations(times, result_data["y"], result_data["vy"], centered=True, as_int=True)

    # Allocate space for the coadds in the results table.  We do this onces because we need rows
    # for entries in the table, but will only fill them in one entry (trajectory) at a time.
    for coadd_type in coadd_types:
        result_data.table[f"coadd_{coadd_type}"] = np.zeros((num_res, width, width), dtype=np.float32)
    for day in days_to_use:
        for coadd_type in coadd_types:
            coadd_str = f"coadd_{coadd_type}{day}"
            result_data.table[coadd_str] = np.zeros((num_res, width, width), dtype=np.float32)

    # Loop through each trajectory generating the coadds.  We extract the stamp stack once
    # for each trajectory and compute all the coadds from that stack.
    to_include = np.full(len(times), True)
    for idx in range(num_res):
        # If we are only using valid observations, retrieve those and filter the day strings.
        if valid_only:
            to_include = result_data["obs_valid"][idx]
        sci_stack = extract_stamp_stack(
            im_stack.sci, xvals[idx, :], yvals[idx, :], radius, to_include=to_include
        )
        sci_stack = np.asanyarray(sci_stack)

        # Only generate the variance stamps if we need them for a weighted co-add.
        if "weighted" in coadd_types:
            var_stack = extract_stamp_stack(
                im_stack.var, xvals[idx, :], yvals[idx, :], radius, to_include=to_include
            )
            var_stack = np.asanyarray(var_stack)

        # Do overall coadds.
        if "mean" in coadd_types:
            result_data[f"coadd_mean"][idx][:, :] = coadd_mean(sci_stack)
        if "median" in coadd_types:
            result_data[f"coadd_median"][idx][:, :] = coadd_median(sci_stack)
        if "sum" in coadd_types:
            result_data[f"coadd_sum"][idx][:, :] = coadd_sum(sci_stack)
        if "weighted" in coadd_types:
            result_data[f"coadd_weighted"][idx][:, :] = coadd_weighted(sci_stack, var_stack)

        # Do nightly coadds if needed.
        for day in days_to_use:
            # Find the valid days that match the current string.
            day_mask = day == day_strs[to_include]
            sci_day = sci_stack[day_mask]

            if "mean" in coadd_types:
                result_data[f"coadd_mean{day}"][idx][:, :] = coadd_mean(sci_day)
            if "median" in coadd_types:
                result_data[f"coadd_median{day}"][idx][:, :] = coadd_median(sci_day)
            if "sum" in coadd_types:
                result_data[f"coadd_sum{day}"][idx][:, :] = coadd_sum(sci_day)
            if "weighted" in coadd_types:
                result_data[f"coadd_weighted{day}"][idx][:, :] = coadd_weighted(
                    sci_day,
                    var_stack[day_mask],
                )

    stamp_timer.stop()


def append_all_stamps(result_data, im_stack, stamp_radius):
    """Get the stamps for the final results from a kbmod search. These are appended
    onto the corresponding entries in a ResultList.

    Parameters
    ----------
    result_data : `Result`
        The current set of results. Modified directly.
    im_stack : `ImageStackPy`
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
    num_times = im_stack.num_times
    times = im_stack.zeroed_times
    if isinstance(im_stack, ImageStackPy):
        sci_data = im_stack.sci
    else:
        raise TypeError("im_stack must be an ImageStackPy")

    # Predict the x and y locations in a giant batch.
    num_res = len(result_data)
    xvals = predict_pixel_locations(times, result_data["x"], result_data["vx"], centered=True, as_int=True)
    yvals = predict_pixel_locations(times, result_data["y"], result_data["vy"], centered=True, as_int=True)

    all_stamps = np.zeros((num_res, num_times, width, width), dtype=np.float32)
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


def filter_stamps_by_cnn(
    result_data, model_path, model_type="resnet18", coadd_type="mean", stamp_radius=10, verbose=False
):
    """Given a set of results data, run the the requested coadded stamps through a
    provided convolutional neural network and assign a new column that contains the
    stamp classification, i.e. whether or not the result passed the CNN filter.

    Parameters
    ----------
    result_data : `Result`
        The current set of results. Modified directly.
    model_path : `str`
        Path to the the pytorch model and weights file.
    coadd_type : `str`
        Which coadd type to use in the filtering. Depends on how the model was trained.
        Default is 'mean', will grab stamps from the 'coadd_mean' column.
    stamp_radius : `int`
        The radius used to generate the stamps. The dimension of the stamps should be
        (stamp_radius * 2) + 1.
    verbose : `bool`
        Verbosity option for the CNN predicition. Off by default.
    """

    coadd_column = f"coadd_{coadd_type}"
    if coadd_column not in result_data.colnames:
        raise ValueError("result_data does not have provided coadd type as a column.")

    stamps = result_data.table[coadd_column].data
    print(stamps.shape)
    stamp_dimm = (stamp_radius * 2) + 1
    stamp_shape = (1, stamp_dimm, stamp_dimm)
    normalized_stamps = _normalize_stamps(stamps, stamp_dimm)
    normalized_stamps = np.expand_dims(normalized_stamps, axis=1)
    print(normalized_stamps.shape)

    model = MODEL_TYPES[model_type](num_classes=2)
    cnn = _KBMLModel(
        model=model,
        weights=model_path,
        shape=stamp_shape,
    )
    # we'll need to check if we have a GPU available.
    if model_path:
        if torch.cuda.is_available():
            cnn.load_state_dict(torch.load(model_path))
        else:
            cnn.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    cnn.eval()

    stamp_tensor = torch.from_numpy(normalized_stamps)

    predictions = cnn(stamp_tensor)

    classifications = []
    for p in predictions.detach().numpy():
        classifications.append(np.argmax(p))

    bool_arr = np.array(classifications) != 0
    result_data.table["cnn_class"] = bool_arr
