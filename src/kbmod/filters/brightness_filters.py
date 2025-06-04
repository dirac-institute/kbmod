import numpy as np

from kbmod.core.stamp_utils import extract_curve_values


def extract_sci_curves(res, stack, append=True):
    """Extract the science pixel curves for each candidate in the results object.

    Parameters
    ----------
    res : `Results``
        The search results object containing trajectories.
    stack : `ImageStack`
        The image stack used for the search, containing the images.
    append : `bool`, optional
        If True, the extracted curves will be appended to the results data.

    Returns
    -------
    sci_curves : `numpy.ndarray`
        An array of shape (num_candidates, num_times) containing the extracted science pixel values
        with NaN for pixels outside the image bounds.
    """
    x_pos = (res["x"][:, np.newaxis] + stack.zeroed_times * res["vx"][:, np.newaxis] + 0.5).astype(int)
    y_pos = (res["y"][:, np.newaxis] + stack.zeroed_times * res["vy"][:, np.newaxis] + 0.5).astype(int)

    sci_curves = extract_curve_values(stack.sci, x_pos, y_pos)
    if append:
        res.table["sci_curves"] = sci_curves
    return sci_curves


def extract_var_curves(res, stack, append=True):
    """Extract the variance pixel curves for each candidate in the results object.

    Parameters
    ----------
    res : `Results``
        The search results object containing trajectories.
    stack : `ImageStack`
        The image stack used for the search, containing the images.
    append : `bool`, optional
        If True, the extracted curves will be appended to the results data.

    Returns
    -------
    var_curves : `numpy.ndarray`
        An array of shape (num_candidates, num_times) containing the extracted variance pixel values
        with NaN for pixels outside the image bounds.
    """
    x_pos = (res["x"][:, np.newaxis] + stack.zeroed_times * res["vx"][:, np.newaxis] + 0.5).astype(int)
    y_pos = (res["y"][:, np.newaxis] + stack.zeroed_times * res["vy"][:, np.newaxis] + 0.5).astype(int)

    var_curves = extract_curve_values(stack.var, x_pos, y_pos)
    if append:
        res.table["var_curves"] = var_curves
    return var_curves
