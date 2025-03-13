"""Utility functions for working with images as numpy arrays."""

import numpy as np

from kbmod.search import (
    ImageStack,
    LayeredImage,
    PSF,
    RawImage,
)


def image_allclose(img_a, img_b, atol=1e-6):
    """Determine whether two images are almost equal, accounting for NO_DATA.

    Parameters
    ----------
    img_a : numpy.ndarray
        The first image.
    img_b : numpy.ndarray
        The second image.
    atol : `float`
        The absolute tolerance of pixel value differences.
        Default: 1e-6

    Returns
    -------
    are_close : `bool`
        Whether the images are close.
    """
    if img_a.shape != img_b.shape:
        return False

    # Check that both images have the same mask of valid pixels.
    valid_a = np.isfinite(img_a)
    valid_b = np.isfinite(img_b)
    if np.any(valid_a != valid_b):
        return False

    # Check the absolute differences of the valid pixels.
    diffs = img_a[valid_a] - img_b[valid_b]
    if np.any(np.abs(diffs) > atol):
        return False

    return True


def extract_sci_images_from_stack(im_stack):
    """Extract the science images in an ImageStack into a single T x H x W numpy array
    where T is the number of times (images), H is the image height in pixels, and W is
    the image width in pixels.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.

    Returns
    -------
    img_array : `np.array`
        The T x H x W numpy array of science data.
    """
    num_images = im_stack.img_count()
    img_array = np.empty((num_images, im_stack.get_height(), im_stack.get_width()))
    for idx in range(num_images):
        img_array[idx, :, :] = im_stack.get_single_image(idx).get_science().image
    return img_array


def extract_var_images_from_stack(im_stack):
    """Extract the variance images in an ImageStack into a single T x H x W numpy array
    where T is the number of times (images), H is the image height in pixels, and W is
    the image width in pixels.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.

    Returns
    -------
    img_array : `np.array`
        The T x H x W numpy array of variance data.
    """
    num_images = im_stack.img_count()
    img_array = np.empty((num_images, im_stack.get_height(), im_stack.get_width()))
    for idx in range(num_images):
        img_array[idx, :, :] = im_stack.get_single_image(idx).get_variance().image
    return img_array


def image_stack_from_components(times, sci, var, mask=None, psfs=None):
    """Construct an ImageStack from the components.

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
        psfs = [PSF() for i in range(num_times)]
    elif len(psfs) != num_times:
        raise ValueError(f"PSF data must have {num_times} entries.")

    # Create the image stack one image at a time.
    im_stack = ImageStack()
    for idx in range(num_times):
        psf = psfs[idx]
        if not isinstance(psfs[idx], PSF):
            psf = PSF(psfs[idx])

        img = LayeredImage(
            sci[idx, :, :].astype(np.single),
            var[idx, :, :].astype(np.single),
            mask[idx, :, :].astype(np.single),
            psf,
            times[idx],
        )

        # force_move destroys img object, but avoids a copy.
        im_stack.append_image(img, force_move=True)

    return im_stack
