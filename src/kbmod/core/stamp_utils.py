"""Functions for generating stamps from a 3D image stack."""

import numpy as np
import torch

from numba import jit, typed

from kbmod.core.image_stack_py import ImageStackPy


# -----------------------------------------------------------------------------
# --- Data Extraction Functions -----------------------------------------------
# -----------------------------------------------------------------------------


def extract_stamp_stack(imgs, x_vals, y_vals, radius, to_include=None):
    """Generate the stack of all stamps for a given trajectory. This can
    be used with any type of stacked images (science, variance, psi, phi).

    Note
    ----
    This is a wrapper around the compiled function that adds some input validation.

    Parameters
    ----------
    imgs : numpy.ndarray or list
        Either a single T x H x W array of image data or a length T list of H x W arrays
        of image data, where T is the number of times, H is the image height,
        and W is the image width.
    x_vals : np.array
        The x values at the center of the stamp. Must be length T.
    y_vals : np.array
        The y values at the center of the stamp. Must be length T.
    radius : int
        The radius of the stamp. Must be >= 1.
    to_include : numpy.ndarray, optional
        A numpy array indicating which images to use. This can either be an array
        of bools, in which case it will be treated as a mask, or a list of integer indices
        for the images to use. If None uses all of images.

    Returns
    -------
    stamp_stack : numpy.ndarray
        If a single array of images is passed in, returns a single T' x (2*R+1) x (2*R+1)
        sized array of stamps. If a list of images is passed in, it returns a length T'
        list of (2*R+1) x (2*R+1) arrays with one for each stamp.
        T' is the number of times to use (from to_include) and R is the stamp radius.
    """
    num_times = len(imgs)
    if radius < 1:
        raise ValueError("Radius must be at least 1.")
    if len(x_vals) != num_times or len(y_vals) != num_times:
        raise ValueError("X and Y values must have the same length as the number of times.")

    # Look at which images to use.
    if to_include is None:
        time_mask = None
    else:
        to_include = np.asarray(to_include)
        if to_include.dtype == bool:
            if len(to_include) != num_times:
                raise ValueError("Time mask must have the same length as the number of times.")
            time_mask = to_include
        elif to_include.dtype == int:
            time_mask = np.full(num_times, False)
            time_mask[to_include] = True

    # Make sure the indices are integers.
    x_vals = np.asarray(x_vals, dtype=int)
    y_vals = np.asarray(y_vals, dtype=int)

    # Call the compiled function.
    if isinstance(imgs, list):
        return _extract_stamp_stack_list(typed.List(imgs), x_vals, y_vals, radius, mask=time_mask)
    else:
        return _extract_stamp_stack_np(imgs, x_vals, y_vals, radius, mask=time_mask)


def extract_curve_values(imgs, x_vals, y_vals):
    """Generate a length T array of image values. This can be used with
    any type of stacked images (science, variance, psi, phi).

    Parameters
    ----------
    imgs : numpy.ndarray or list of numpy.ndarray
        A single T x H x W array of image data, where T is the number of times,
        H is the image height, and W is the image width.
    x_vals : np.array
        The x values at the center of the stamp. Must be length T.
    y_vals : np.array
        The y values at the center of the stamp. Must be length T.

    Returns
    -------
    values : numpy.ndarray
        A length T array where T is the number of times.
    """
    if len(imgs.shape) != 3:
        raise ValueError("Image data must have 3 dimensions.")
    if len(x_vals) != imgs.shape[0] or len(y_vals) != imgs.shape[0]:
        raise ValueError("X and Y values must have the same length as the number of times.")

    # Make sure the indices are integers.
    x_vals = np.asarray(x_vals, dtype=int)
    y_vals = np.asarray(y_vals, dtype=int)

    # Extract the values of the pixels that fall within the images.
    values = _extract_curve_values(imgs, x_vals, y_vals)
    return values


def create_stamps_from_image_stack_xy(stack, radius, xvals, yvals, to_include=None):
    """Create a vector of stamps centered on the predicted position
    of an Trajectory at different times.

    Parameters
    ----------
    stack : `ImageStackPy`
        The stack of images to use.
    xvals : `list` of `int`
        The x-coordinate of the stamp center at each time.
    yvals : `list` of `int`
        The y-coordinate of the stamp center at each time.
    radius : `int`
        The stamp radius in pixels. The total stamp width = 2*radius+1.
    to_include : numpy.ndarray, optional
        A numpy array indicating which images to use. This can either be an array
        of bools, in which case it will be treated as a mask, or a list of integer indices
        for the images to use. If None uses all of images.

    Returns
    -------
    `list` of `np.ndarray`
        The stamps.
    """
    if isinstance(stack, ImageStackPy):
        img_data = stack.sci
    else:
        raise ValueError("Invalid image stack type.")

    # Create the stamps.
    stamps = extract_stamp_stack(img_data, xvals, yvals, radius, to_include=to_include)
    return stamps


def create_stamps_from_image_stack(stack, trj, radius, to_include=None):
    """Create a vector of stamps centered on the predicted position
    of an Trajectory at different times.

    Parameters
    ----------
    stack : `ImageStackPy`
        The stack of images to use.
    trj : Trajectory-like
        The trajectory to project to each time. This can be any object with attributes
        x, y, vx, and vy.
    radius : `int`
        The stamp radius in pixels. The total stamp width = 2*radius+1.
    to_include : numpy.ndarray, optional
        A numpy array indicating which images to use. This can either be an array
        of bools, in which case it will be treated as a mask, or a list of integer indices
        for the images to use. If None uses all of images.

    Returns
    -------
    `list` of `np.ndarray`
        The stamps.
    """
    # Predict the Trajectory's position.
    xvals = (trj.x + stack.zeroed_times * trj.vx + 0.5).astype(int)
    yvals = (trj.y + stack.zeroed_times * trj.vy + 0.5).astype(int)

    # Create the stamps.
    return create_stamps_from_image_stack_xy(stack, radius, xvals, yvals, to_include=to_include)


# -----------------------------------------------------------------------------
# --- Coadd Functions ---------------------------------------------------------
# -----------------------------------------------------------------------------


def _mask_all_nans(stack):
    """Mask out any pixels with NaNs at all times.

    Parameters
    ----------
    stack : `numpy.ndarray`
        A T x H x W sized array of data to coadd where T is the number
        of times, H is the image height, and W is the image width.

    Returns
    -------
    stack : `numpy.ndarray`
        A new copy of the input stack with any columns that are all NaNs set to 0.0.
    """
    stack = np.asarray(stack)
    n_times, height, width = stack.shape

    # Find pixels that are NaN at all times.
    no_pixel_valid = np.all(np.isnan(stack), axis=0)
    if np.any(no_pixel_valid):
        pixel_mask = np.tile(no_pixel_valid.flatten(), n_times).reshape(n_times, height, width)

        stack = stack.copy()
        stack[pixel_mask] = 0.0
    return stack


def coadd_sum(stack):
    """Generate the sum of all the stamps in the stack, ignoring NaNs.

    Parameters
    ----------
    stack : `numpy.ndarray`
        A T x H x W sized array of data to coadd where T is the number
        of times, H is the image height, and W is the image width.

    Returns
    -------
    coadd : `numpy.ndarray`
        A H x W sized array of the coadded stamp.
    """
    return np.nansum(stack, axis=0)


def coadd_mean(stack):
    """Generate the mean of all the stamps in the stack, ignoring NaNs.

    Parameters
    ----------
    stack : `numpy.ndarray`
        A T x H x W sized array of data to coadd where T is the number
        of times, H is the image height, and W is the image width.

    Returns
    -------
    coadd : `numpy.ndarray`
        A H x W sized array of the coadded stamp.
    """
    if stack.shape[0] == 0:
        return np.zeros((stack.shape[1], stack.shape[2]), dtype=stack.dtype)
    stack = _mask_all_nans(stack)
    return np.nanmean(stack, axis=0)


def coadd_median(stack, device=None):
    """Generate the median of all the stamps in the stack, ignoring NaNs.

    Parameters
    ----------
    stack : `numpy.ndarray`
        A T x H x W sized array of data to coadd where T is the number
        of times, H is the image height, and W is the image width.
    device : `torch.device`, optional
        The device to use for the convolution.
        If None, the default device is used.

    Returns
    -------
    coadd : `numpy.ndarray`
        A H x W sized array of the coadded stamp.
    """
    if stack.shape[0] == 0:
        return np.zeros((stack.shape[1], stack.shape[2]), dtype=stack.dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stack_tensor = torch.tensor(stack, device=device)
    coadd_tensor, _ = torch.nanmedian(stack_tensor, dim=0)
    coadd_tensor[torch.isnan(coadd_tensor)] = 0.0
    return coadd_tensor.cpu().numpy()


def coadd_weighted(stack, var_stack):
    """Generate the variance weighted mean of all the stamps in the stack.

    Parameters
    ----------
    stack : `numpy.ndarray`
        A T x H x W sized array of data to coadd where T is the number
        of times, H is the image height, and W is the image width.
    var_stack : `numpy.ndarray`
        A T x H x W sized array of variance values corresponding to the
        science stack.

    Returns
    -------
    coadd : `numpy.ndarray`
        A H x W sized array of the coadded stamp.
    """
    stack = _mask_all_nans(stack)

    # Compute the pixels that are valid to use in the variance weighted computation.
    pix_valid = ~(np.isnan(stack) | np.isnan(var_stack) | (var_stack == 0.0))

    # Compute the weighted science values and the weights.
    n_times, height, width = stack.shape
    weights = np.zeros((n_times, height, width))
    weights[pix_valid] = 1.0 / var_stack[pix_valid]

    # Compute the variance weighted values of the science pixels.
    weighted_sci = np.zeros((n_times, height, width))
    weighted_sci[pix_valid] = stack[pix_valid] * weights[pix_valid]
    weighted_sum = np.sum(weighted_sci, axis=0)

    # Compute the scaling factor (sum of the weights) for each pixel.
    # If a pixel has no data, then use a large scaling factor to avoid divide by zero.
    sum_of_weights = np.sum(weights, axis=0)
    sum_of_weights[sum_of_weights == 0.0] = 1e24

    return weighted_sum / sum_of_weights


# -----------------------------------------------------------------------------
# --- Compiled Functions ------------------------------------------------------
# -----------------------------------------------------------------------------


@jit(nopython=True)
def extract_stamp(img, x_val, y_val, radius):
    """Generate a single stamp as an numpy array from a given time index
    and centered on a given x, y location.

    Parameters
    ----------
    img : `np.array`
        The H x W array of image data.
    x_val : `int`
        The x value corresponding to the center pixel of the stamp.
    y_val : `int`
        The y value corresponding to the center pixel of the stamp..
    radius : `int`
        The radius of the stamp (in pixels). Must be >= 1.

    Returns
    -------
    stamp : numpy.ndarray
        A square (2 * radius + 1, 2 * radius + 1) matrix representing
        the stamp with NaNs anywhere there is no data.
    """
    (img_height, img_width) = img.shape

    # Compute the start and end x locations in the full image [x_img_s, x_img_e] and the
    # corresponding bounds in the stamp [x_stp_s, x_stp_e].
    x_img_s = 0 if x_val - radius < 0 else x_val - radius
    x_img_e = img_width if x_val + radius + 1 >= img_width else x_val + radius + 1
    x_width = x_img_e - x_img_s
    x_stp_s = x_img_s - (x_val - radius)
    x_stp_e = x_stp_s + x_width

    # Compute the start and end y locations in the full image [y_img_s, y_img_e] and the
    # corresponding bounds in the stamp [y_stp_s, y_stp_e].
    y_img_s = 0 if y_val - radius < 0 else y_val - radius
    y_img_e = img_height if y_val + radius + 1 >= img_height else y_val + radius + 1
    y_width = y_img_e - y_img_s
    y_stp_s = y_img_s - (y_val - radius)
    y_stp_e = y_stp_s + y_width

    # Create the stamp. Start with an array of NaN and then fill in whatever we cut
    # out of the image. Don't fill in anything if the stamp is completely off the image.
    stamp = np.full((2 * radius + 1, 2 * radius + 1), np.nan)
    if y_img_s <= y_img_e and x_img_s <= x_img_e:
        stamp[y_stp_s:y_stp_e, x_stp_s:x_stp_e] = img[y_img_s:y_img_e, x_img_s:x_img_e]
    return stamp


# Note that batching this over trajectories using a double loop (trjs and times)
# and writing into a single array does not help much. This is surprising because
# I would have expected compiling the outer loop to make a huge difference.
@jit(nopython=True)
def _extract_stamp_stack_np(imgs, x_vals, y_vals, radius, mask=None):
    """Generate a T x S x S sized array of stamps where T is the number
    of times to use and S is the stamp width (2 * radius + 1).

    Parameters
    ----------
    imgs : numpy.ndarray
        A single T x H x W array of image data, where T is the number of times,
        H is the image height, and W is the image width.
    x_vals : np.array of int
        The x values at the center of the stamp. Must be length T.
    y_vals : np.array of int
        The y values at the center of the stamp. Must be length T.
    radius : int
        The radius of the stamp. Must be >= 1.
    mask : numpy.ndarray, optional
        A numpy array of bools indicating which images to use. If None,
        uses all of the images.

    Returns
    -------
    stamp_stack : numpy.ndarray
        A T x (2*R+1) x (2*R+1) sized array where T is the number of times and R is
        the stamp radius.
    """
    num_times = len(imgs)
    num_stamps = num_times if mask is None else np.count_nonzero(mask)
    stamp_stack = np.full((num_stamps, 2 * radius + 1, 2 * radius + 1), np.nan)

    # Fill in each unmasked time step.
    current = 0
    for idx in range(num_times):
        if mask is None or mask[idx]:
            stamp_stack[current] = extract_stamp(imgs[idx], x_vals[idx], y_vals[idx], radius)
            current += 1
    return stamp_stack


@jit(nopython=True)
def _extract_stamp_stack_list(imgs, x_vals, y_vals, radius, mask=None):
    """Generate a lenght T list of S x S sized array of stamps where T is the number
    of times to use and S is the stamp width (2 * radius + 1).

    Parameters
    ----------
    imgs : numpy.ndarray or list of numpy.ndarray
        A list of T arrays H x W  of image data where T is the number
        of times, H is the image height, and W is the image width.
    x_vals : np.array of int
        The x values at the center of the stamp. Must be length T.
    y_vals : np.array of int
        The y values at the center of the stamp. Must be length T.
    radius : int
        The radius of the stamp. Must be >= 1.
    mask : numpy.ndarray, optional
        A numpy array of bools indicating which images to use. If None,
        uses all of the images.

    Returns
    -------
    stamp_stack : list
        A length T list of S x S arrays where T is the number of times to use
        and S is the stamp width (2 * radius + 1).
    """
    num_times = len(imgs)
    num_stamps = num_times if mask is None else np.count_nonzero(mask)
    stamp_stack = [np.full((2 * radius + 1, 2 * radius + 1), np.nan) for i in range(num_stamps)]

    # Fill in each unmasked time step.
    current = 0
    for idx in range(num_times):
        if mask is None or mask[idx]:
            stamp_stack[current] = extract_stamp(imgs[idx], x_vals[idx], y_vals[idx], radius)
            current += 1
    return stamp_stack


@jit(nopython=True)
def _extract_curve_values(imgs, x_vals, y_vals):
    """Generate a length T array of image values. This can be used with
    any type of stacked images (science, variance, psi, phi).

    Parameters
    ----------
    imgs : numpy.ndarray or list of numpy.ndarray
        A single T x H x W array of image data, where T is the number of times,
        H is the image height, and W is the image width.
    x_vals : np.array of int
        The x values at the center of the stamp. Must be length T.
    y_vals : np.array of int
        The y values at the center of the stamp. Must be length T.

    Returns
    -------
    values : numpy.ndarray
        A length T array where T is the number of times.
    """
    (num_times, width, height) = imgs.shape

    # Extract the values of the pixels that fall within the images.
    values = np.full(num_times, np.nan)
    for idx in range(num_times):
        if x_vals[idx] >= 0 and x_vals[idx] < width and y_vals[idx] >= 0 and y_vals[idx] < height:
            values[idx] = imgs[idx, y_vals[idx], x_vals[idx]]
    return values
