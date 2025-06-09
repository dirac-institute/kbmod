import numpy as np
import torch

from kbmod.core.stamp_utils import extract_curve_values


def extract_sci_var_curves(res, stack, keep_nans=True, append=True):
    """Extract the science and variance pixel curves for each candidate in the results object.

    Parameters
    ----------
    res : `Results``
        The search results object containing trajectories.
    stack : `ImageStack`
        The image stack used for the search, containing the images.
    keep_nans : `bool`, optional
        If True, the extracted curves will keep NaN values for masked pixels (including out of bounds).
        If False, those values will be replaced with 0 for science and 1 for variance.
    append : `bool`, optional
        If True, the extracted curves will be appended to the results data.
        Default: True

    Returns
    -------
    sci_curves : `numpy.ndarray`
        An array of shape (num_candidates, num_times) containing the extracted science pixel values
        with the mask value (default=NaN) for pixels outside the image bounds.
    """
    # Compute the predicted x and y positions for each candidate at each time step.
    # The +0.5 is to account for the pixel center offset.
    x_pos = (res["x"][:, np.newaxis] + stack.zeroed_times * res["vx"][:, np.newaxis] + 0.5).astype(int)
    y_pos = (res["y"][:, np.newaxis] + stack.zeroed_times * res["vy"][:, np.newaxis] + 0.5).astype(int)

    # Extract the science and variance pixels values and replace the masked values if needed.
    sci_curves = extract_curve_values(stack.sci, x_pos, y_pos)
    var_curves = extract_curve_values(stack.var, x_pos, y_pos)
    if not keep_nans:
        sci_curves[np.isnan(sci_curves)] = 0
        var_curves[np.isnan(var_curves)] = 1

    # Append to extracted curves to the results.
    if append:
        res.table["sci_curve"] = sci_curves
        res.table["var_curve"] = var_curves

    return sci_curves, var_curves
