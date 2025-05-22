"""Utility functions for working with images as numpy arrays."""

import logging
import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.core.stamp_utils import extract_stamp_stack

logger = logging.getLogger(__name__)


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
