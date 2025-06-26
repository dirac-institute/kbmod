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
        var_curves[np.isnan(var_curves)] = 1e10

    # Append to extracted curves to the results.
    if append:
        res.table["sci_curve"] = sci_curves
        res.table["var_curve"] = var_curves

    return sci_curves, var_curves


def score_brightness_candidates(sci_curves, var_curves, brightness_candidates):
    """Score the likelihood of the data given different brightness candidates in
    the context of each pixel's science and variance curves.  The code uses a variance-weighted,
    squared distance metric that is proportional to the likelihood. Lower scores
    indicate a better fit to the data.

    Parameters
    ----------
    sci_curves : `np.ndarray`
        A R x T array of science pixel values, where R is the number of results
        and T is the number of time steps.
    var_curves : `np.ndarray`
        A R x T array of variance pixel values, where R is the number of results
        and T is the number of time steps.
    brightness_candidates : `np.ndarray`
        A 1-dimensional length C array (same candidates for all results) or a 2-dimensional
        R x C array of brightness candidates (customized list for each result),
        where C is the number of brightness candidates and R is the number of results.

    Returns
    -------
    scores : `np.ndarray`
        An R x C array of scores for each (brightness candidate, result) pair, where
        lower scores indicate a better fit to the data.
    """
    # Use a GPU if one is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Make the curves into R x T x 1 tensors, where R is the number of results
    # and T is the number of time steps. The first dimension will be used for batch
    # processing against the brightness candidates.
    sci_t = torch.tensor(sci_curves, dtype=torch.float32, device=device).unsqueeze(2)
    var_t = torch.tensor(var_curves, dtype=torch.float32, device=device).unsqueeze(2)

    # Mask out the NaN values or variance=0 values.
    masked = torch.isnan(sci_t) | (var_t <= 0) | torch.isnan(var_t)
    sci_t[masked] = 0
    var_t[masked] = 1e10  # effectively ignore masked pixels

    if brightness_candidates.ndim == 1:
        # Convert the brightness candidates to a 1 x 1 x C tensor.
        candidates_t = (
            torch.tensor(
                brightness_candidates,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
    elif brightness_candidates.ndim == 2:
        # Convert the brightness candidates to a 1 x R x C tensor.
        if len(brightness_candidates) != len(sci_curves):
            raise ValueError("Brightness candidates must have the same number of results as sci_curves.")
        candidates_t = torch.tensor(
            brightness_candidates,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

    # Compute the variance-weighted squared distance for each brightness candidate.
    score = torch.sum((sci_t - candidates_t) ** 2 / var_t, dim=1)
    return score.cpu().numpy()


def local_search_brightness(
    sci_curves,
    var_curves,
    brightness=None,
    offsets=[0.5, 0.75, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.25, 1.5],
):
    """Perform a local search for the best brightness value given the science and variance curves
    for each result.

    Note that the best brightness value found will not necessarily match the true flux,
    because this function only considers the flux in a single pixel and the brightness
    has been spread out by the PSF. Instead this should be used to determine if the
    brightness value is close to what was estimate by the search.

    Parameters
    ----------
    sci_curves : `np.ndarray`
        A R x T array of science pixel values, where R is the number of results
        and T is the number of time steps.
    var_curves : `np.ndarray`
        A R x T array of variance pixel values, where R is the number of results
        and T is the number of time steps.
    brightness : `np.ndarray`
        A length R array of brightness values to use as the center of the local search. If None,
        uses the median brightness of the science pixels.
        Default: None
    offsets : `list` of `float`
        A list of multiplicative offsets to apply to the brightness value for the local search.
        Default: [0.5, 0.75, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.25, 1.5]

    Returns
    -------
    best_brightness : `np.ndarray`
        A length R array of the best brightness values for each result.
    best_idx : `np.ndarray`
        A length R array of the indices of the best brightness values in the offsets list.
    """
    num_results = sci_curves.shape[0]
    if sci_curves.shape != var_curves.shape:
        raise ValueError(
            f"var_curves and sci_curves must have the same shape: {var_curves.shape} vs {sci_curves.shape}."
        )

    # If brightness is not provided, use the median brightness of the science pixels.
    if brightness is None:
        brightness = np.nanmedian(sci_curves, axis=1)
    elif len(brightness) != num_results:
        raise ValueError(f"Brightness must have the same number of elements as sci_curves ({num_results}).")
    else:
        brightness = np.asanyarray(brightness)

    # Compute the score for each pair of (brightness candidate, result).
    brightness_candidates = brightness[:, np.newaxis] * np.asanyarray(offsets)[np.newaxis, :]
    scores = score_brightness_candidates(sci_curves, var_curves, brightness_candidates)
    best_idx = np.argmin(scores, axis=1)
    best_brightness = brightness_candidates[np.arange(num_results), best_idx]

    return best_brightness, best_idx


def apply_brightness_search_filter(
    results,
    im_stack,
    offsets=[0.5, 0.9, 1.0, 1.1, 1.5],
    save_curves=True,
):
    """Apply a filter that computes the likelihood of a trajectory assuming different underlying
    brightness values. Results are filtered if their best brightness value is one of the extreme
    offsets, which indicates that the brightness is likely not well estimated due to an outlier.

    This filter is adapted from the brightness search filter described in Wesley Fraser's pkmod:
    https://github.com/fraserw/pkbmod

    Parameters
    ----------
    results : `Results`
        The results object containing the trajectories to filter.
        This data gets modified directly by the filtering.
    im_stack : `ImageStack`
        The image stack used for the search, containing the images.
    offsets : `list` of `float`, optional
        A list of multiplicative offsets to apply to the brightness value for the local search.
        Default: [0.5, 0.9, 1.0, 1.1, 1.5]
    save_curves : `bool`, optional
        If True, the science and variance curves will be saved to the results object.
        Default: True.
    """
    # If we do not have the science and the variance curves in the results,
    # extract them from the image stack.
    if "sci_curve" in results.colnames and "var_curve" in results.colnames:
        sci_curves = results["sci_curve"]
        var_curves = results["var_curve"]
    else:
        sci_curves, var_curves = extract_sci_var_curves(results, im_stack, append=save_curves)

    # Check for better brightness matches using multipliers around the estimated flux.
    _, best_idx = local_search_brightness(
        sci_curves,
        var_curves,
        brightness=results["flux"],
        offsets=offsets,
    )

    keep_mask = np.isin(best_idx, [0, len(offsets) - 1], invert=True)
    results.filter_rows(keep_mask, "local_brightness_search")
