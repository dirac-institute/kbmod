"""Utility functions for working with images as numpy arrays."""

import numpy as np

from kbmod.search import (
    ImageStack,
    LayeredImage,
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
