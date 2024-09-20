"""A collection of utility functions for working with files in KBMOD."""

import csv
import re
from collections import OrderedDict
from itertools import product
from math import copysign
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import *
from astropy.io import fits
from astropy.time import Time

import kbmod.search as kb
from kbmod.search import LayeredImage
from kbmod.trajectory_utils import trajectory_from_np_object


def load_deccam_layered_image(filename, psf):
    """Load a layered image from the legacy deccam format.

    Parameters
    ----------
    filename : `str`
        The name of the file to load.
    psf : `PSF`
        The PSF to use for the image.

    Returns
    -------
    img : `LayeredImage`
        The loaded image.

    Raises
    ------
    Raises a ``FileNotFoundError`` if the file does not exist.
    Raises a ``ValueError`` if any of the validation checks fail.
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"{filename} not found")

    img = None
    with fits.open(filename) as hdul:
        if len(hdul) < 4:
            raise ValueError("Not enough extensions for legacy deccam format")

        # Extract the obstime trying from a few keys and a few extensions.
        obstime = -1.0
        for key, ext in product(["MJD", "DATE-AVG", "MJD-OBS"], [0, 1]):
            if key in hdul[ext].header:
                value = hdul[ext].header[key]
                if type(value) is float:
                    obstime = value
                    break
                if type(value) is str:
                    timesys = hdul[ext].header.get("TIMESYS", "UTC").lower()
                    obstime = Time(value, scale=timesys).mjd
                    break

        img = LayeredImage(
            hdul[1].data.astype(np.float32),  # Science
            hdul[3].data.astype(np.float32),  # Variance
            hdul[2].data.astype(np.float32),  # Mask
            psf,
            obstime,
        )

    return img
