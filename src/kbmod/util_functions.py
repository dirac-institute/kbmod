"""General purpose utility functions."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from itertools import product

from kbmod.search import LayeredImage


def get_matched_obstimes(obs_times, query_times, threshold=0.0007):
    """Given a list of times, returns the indices of images that are close enough to the query times.

    Parameters
    ----------
    obs_times : list-like
        The times from the data set.
    query_times : list-like
        The query times.
    threshold : float
        The match threshold (in days)
        Default: 0.0007 = 1 minute

    Returns
    -------
    match_indices : np.array
        The matching index for each obs time. Set to -1 if there is not obstime within
        the given threshold.
    """
    # Create a version of the data times bounded by -inf and inf.
    all_times = np.insert(obs_times, [0, len(obs_times)], [-np.inf, np.inf])

    # Find each query time's insertion point in the sorted array.  Because we inserted
    # -inf and inf we have 0 < sorted_inds <= len(all_times).
    sorted_inds = np.searchsorted(all_times, query_times, side="left")
    right_dist = np.abs(all_times[sorted_inds] - query_times)
    left_dist = np.abs(all_times[sorted_inds - 1] - query_times)

    min_dist = np.where(left_dist > right_dist, right_dist, left_dist)
    min_inds = np.where(left_dist > right_dist, sorted_inds, sorted_inds - 1)

    # Filter out matches that exceed the threshold.
    # Shift back to account for the -inf inserted at the start.
    min_inds = np.where(min_dist <= threshold, min_inds - 1, -1)

    return min_inds


def mjd_to_day(mjd):
    """Takes an mjd and converts it into a day in calendar date format.

    Parameters
    ----------
    mjd : `float`
        mjd format date.

    Returns
    ----------
    A `str` with a calendar date, in the format YYYY-MM-DD.
    e.g., mjd=60000 -> '2023-02-25'
    """
    return Time(mjd, format="mjd").strftime("%Y-%m-%d")


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
