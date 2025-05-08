"""Utility functions for working with images as numpy arrays."""

import logging
import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.core.stamp_utils import extract_stamp_stack
from kbmod.search import ImageStack, LayeredImage

logger = logging.getLogger(__name__)


def extract_sci_images_from_stack(im_stack):
    """Extract the science images in an ImageStack into a single T x H x W numpy array
    where T is the number of times (images), H is the image height in pixels, and W is
    the image width in pixels.

    This is useful when converting from an ImageStack (C++) to an ImageStackPy (Python).

    Parameters
    ----------
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.

    Returns
    -------
    img_array : `np.array`
        The T x H x W numpy array of science data.
    """
    num_images = im_stack.num_times
    img_array = np.empty((num_images, im_stack.height, im_stack.width))
    for idx in range(num_images):
        img_array[idx, :, :] = im_stack.get_single_image(idx).sci
    return img_array


def extract_var_images_from_stack(im_stack):
    """Extract the variance images in an ImageStack into a single T x H x W numpy array
    where T is the number of times (images), H is the image height in pixels, and W is
    the image width in pixels.

    This is useful when converting from an ImageStack (C++) to an ImageStackPy (Python).

    Parameters
    ----------
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.

    Returns
    -------
    img_array : `np.array`
        The T x H x W numpy array of variance data.
    """
    num_images = im_stack.num_times
    img_array = np.empty((num_images, im_stack.height, im_stack.width))
    for idx in range(num_images):
        img_array[idx, :, :] = im_stack.get_single_image(idx).var
    return img_array


def image_stack_from_components(times, sci, var, mask=None, psfs=None):
    """Construct an ImageStack from the components.

    This is useful when converting from an ImageStackPy (Python) to an ImageStack (C++).

    Attributes
    ----------
    times : np.array
        The length T array of variance data.
    sci : np.array
        The T x H x W array of science data.
    var : np.array
        The T x H x W array of variance data.
    mask : np.array, optional
        The T x H x W array of mask data.
    psfs : list, optional
        The length T array of PSF information.

    Parameters
    ----------
    im_stack : `ImageStack`
        The image data for KBMOD.
    """
    if times is None or len(times) == 0:
        raise ValueError("Cannot create an ImageStack with no times.")
    num_times = len(times)

    sci = np.asarray(sci, dtype=float)
    var = np.asarray(var, dtype=float)
    if len(sci.shape) != 3:
        raise ValueError("3d (T x H x W) numpy array of science data required to build stack.")
    if sci.shape[0] != num_times:
        raise ValueError(f"Science data must have {self.num_times} images.")
    if sci.shape != var.shape:
        raise ValueError("Science and variance data must have the same shape.")

    if mask is None:
        mask = np.zeros_like(sci)
    else:
        mask = np.asarray(mask, dtype=float)
        if sci.shape != mask.shape:
            raise ValueError("Science and mask data must have the same shape.")

    # Checks (and creates defaults) for the PSF input.
    if psfs is None:
        psfs = [np.ones((1, 1)) for i in range(num_times)]
    elif len(psfs) != num_times:
        raise ValueError(f"PSF data must have {num_times} entries.")

    # Create the image stack one image at a time.
    im_stack = ImageStack()
    for idx in range(num_times):
        img = LayeredImage(
            sci[idx, :, :].astype(np.single),
            var[idx, :, :].astype(np.single),
            mask[idx, :, :].astype(np.single),
            psfs[idx],
            times[idx],
        )

        # force_move destroys img object, but avoids a copy.
        im_stack.append_image(img, force_move=True)

    return im_stack


def create_stamps_from_image_stack_xy(stack, radius, xvals, yvals, to_include=None):
    """Create a vector of stamps centered on the predicted position
    of an Trajectory at different times.

    Parameters
    ----------
    stack : `ImageStack` or `ImageStackPy`
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
    num_times = stack.num_times

    # Copy the references to the raw image data.
    img_data = [stack.get_single_image(i).sci for i in range(num_times)]

    # Create the stamps.
    stamps = extract_stamp_stack(img_data, xvals, yvals, radius, to_include=to_include)
    return stamps


def create_stamps_from_image_stack(stack, trj, radius, to_include=None):
    """Create a vector of stamps centered on the predicted position
    of an Trajectory at different times.

    Parameters
    ----------
    stack : `ImageStack` or `ImageStackPy`
        The stack of images to use.
    trj : `Trajectory`
        The trajectory to project to each time.
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
    times = np.asarray(stack.zeroed_times)  # linear cost
    xvals = (trj.x + times * trj.vx + 0.5).astype(int)
    yvals = (trj.y + times * trj.vy + 0.5).astype(int)

    # Create the stamps.
    return create_stamps_from_image_stack_xy(stack, radius, xvals, yvals, to_include=to_include)


def _im_stack_validation_error(msg, warn_only):
    """Raise an error or warning based on the warn_only flag.

    Parameters
    ----------
    msg : `str`
        The message to display.
    warn_only : `bool`
        Display a warning instead of raising an exception.
    """
    if warn_only:
        logger.warning(f"WARNING: {msg}")
    else:
        raise ValueError(msg)


def count_valid_images(im_stack, masked_fraction=0.5):
    """Count the number of images in the ImageStack whose fraction
    of masked pixels is below the given threshold. Used for checking
    num_obs.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images to validate.
    masked_fraction: `float`
        The maximum fraction of masked pixels allowed.
        Default: 0.9

    Returns
    -------
    count : `int`
        The count of images with below the threshold fraction of masked pixels.
    """
    total_pixels = im_stack.height * im_stack.width
    if total_pixels == 0 or im_stack.num_times == 0:
        return 0

    valid_count = 0
    for idx in range(im_stack.num_times):
        img = im_stack.get_single_image(idx)
        sci = img.sci
        var = img.var
        mask = img.mask

        # Check for masked pixels.
        is_masked = np.isnan(sci) | np.isnan(var) | (mask != 0) | (var <= 0)
        percent_masked = np.count_nonzero(is_masked) / total_pixels
        if percent_masked <= masked_fraction:
            valid_count += 1

    return valid_count


def validate_image_stack(
    im_stack,
    masked_fraction=0.5,
    min_flux=-1e8,
    max_flux=1e8,
    min_var=1e-20,
    max_var=1e8,
    warn_only=True,
):
    """Run basic validation checks on an image stack.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images to validate.
    masked_fraction: `float`
        The maximum fraction of masked pixels allowed.
        Default: 0.5
    min_flux : `float`
        The minimum flux value allowed.
        Default: -1e8
    max_flux : `float`
        The maximum flux value allowed.
        Default: 1e8
    min_var : `float`
        The minimum variance value allowed.
        Default: -1e8
    max_var : `float`
        The maximum variance value allowed.
        Default: 1e-20 (no zero or negative variance)
    warn_only : `bool`
        Display a warning instead of raising an exception.
        Default: True
    """
    is_valid = True

    total_pixels = im_stack.height * im_stack.width
    if total_pixels == 0 or im_stack.num_times == 0:
        _im_stack_validation_error("Image stack is empty.", warn_only)
        return False

    for idx in range(im_stack.num_times):
        img = im_stack.get_single_image(idx)
        sci = img.sci
        var = img.var
        mask = img.mask

        # Check for masked pixels.
        is_masked = np.isnan(sci) | np.isnan(var) | (mask != 0) | (var <= 0)
        percent_masked = np.count_nonzero(is_masked) / total_pixels
        if percent_masked > masked_fraction:
            _im_stack_validation_error(
                f"Image {idx} has {percent_masked * 100.0} percent masked pixels.",
                warn_only,
            )
            is_valid = False

        # Check for valid flux and variance values.
        if np.nanmin(sci) < min_flux:
            _im_stack_validation_error(
                f"Image {idx} has invalid flux values: {np.nanmin(sci)} < {min_flux}",
                warn_only,
            )
            is_valid = False
        if np.nanmax(sci) > max_flux:
            _im_stack_validation_error(
                f"Image {idx} has invalid flux values: {np.nanmax(sci)} > {max_flux}",
                warn_only,
            )
            is_valid = False
        if np.nanmin(var) < min_var:
            _im_stack_validation_error(
                f"Image {idx} has invalid flux values: {np.nanmin(var)} < {min_var}",
                warn_only,
            )
            is_valid = False
        if np.nanmax(var) > max_var:
            _im_stack_validation_error(
                f"Image {idx} has invalid flux values: {np.nanmax(var)} > {max_var}",
                warn_only,
            )
            is_valid = False

    return is_valid


def stat_image_stack(im_stack):
    """Compute the basic statistics of an image stack and display in a table.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images to analyze.
    """
    total_pixels = im_stack.height * im_stack.width
    num_times = im_stack.num_times

    print("Image Stack Statistics:")
    print(f"  Image Count: {num_times}")
    print(f"  Image Size: {im_stack.height} x {im_stack.width} = {total_pixels}")

    print(
        "+------+------------+------------+------------+------------+----------+----------+----------+--------+"
    )
    print(
        "|  idx |     Time   |  Flux Min  |  Flux Max  |  Flux Mean |  Var Min |  Var Max | Var Mean | Masked |"
    )
    print(
        "+------+------------+------------+------------+------------+----------+----------+----------+--------+"
    )

    for idx in range(num_times):
        img = im_stack.get_single_image(idx)
        sci = img.sci
        var = img.var
        mask = img.mask

        # Count the masked pixels.
        is_masked = np.isnan(sci) | np.isnan(var) | (mask != 0) | (var <= 0)
        percent_masked = (np.count_nonzero(is_masked) / total_pixels) * 100.0

        # Compute the basic statistics.
        flux_min = np.nanmin(sci)
        flux_max = np.nanmax(sci)
        flux_mean = np.nanmean(sci)
        var_min = np.nanmin(var)
        var_max = np.nanmax(var)
        var_mean = np.nanmean(var)

        print(
            f"| {idx:4d} | {img.time:10.3f} | {flux_min:10.2f} | {flux_max:10.2f} | {flux_mean:10.2f} "
            f"| {var_min:8.2f} | {var_max:8.2f} | {var_mean:8.2f} | {percent_masked:6.2f} |"
        )
        print(
            "+------+------------+------------+------------+------------+----------+----------+----------+--------+"
        )
