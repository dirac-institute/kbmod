"""General purpose utility functions."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from itertools import product
import pandas as pd

from kbmod.core.image_stack_py import LayeredImagePy


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
    -------
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
    psf : `np.ndarray`, optional
        The PSF to use for the image.

    Returns
    -------
    img : `LayeredImagePy`
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

        img = LayeredImagePy(
            hdul[1].data.astype(np.float32),  # Science
            hdul[3].data.astype(np.float32),  # Variance
            mask=hdul[2].data.astype(np.float32),  # Mask
            time=obstime,
            psf=psf,
        )

    return img


def get_unique_obstimes(all_obstimes):
    """Get the unique observation times and their indices.
    Used to group observations for mosaicking.

    Parameters
    ----------
    all_obstimes : `np.ndarray`
        The array of observation times.

    Returns
    -------
    unique_obstimes : `np.ndarray`
        The unique observation times.
    unique_indices : `list`
        A list of lists, where each sublist contains the indices of the grouping.
    """
    unique_obstimes = np.unique(all_obstimes)
    unique_indices = [list(np.where(all_obstimes == time)[0]) for time in unique_obstimes]
    return unique_obstimes, unique_indices


def get_magnitude(flux, zero_point):
    """Convert a flux value to a magnitude using the zero point.

    Parameters
    ----------
    flux : `float`
        The flux value to convert.
    zero_point : `float`
        The zero point of the observations.

    Returns
    -------
    mag : `float`
        The calculated magnitude.
    """
    mag = -2.5 * np.log10(flux) + zero_point
    return mag


def unravel_results(results, image_collection, obscode="X05", batch_id=None):
    """Take a results file and transform it into a table of individual observations.

    Parameters
    ----------
    results : `kbmod.results.Results`
        The results.
    image_collection : `kbmod.image_collection.ImageCollection`
        The image collection containing the images used in the results.
    obscode : `str`, optional
        The observatory code to use for the observations.
        Default: "X05" (LSST).
    batch_id : `str`, optional
        The batch ID to use for this result set.
        individual observation ids will be in the format of
        "{batch_id}-{result #}-{observation #}".

    Returns
    -------
    final_df : `pandas.DataFrame`
        A DataFrame containing the individual observations with columns:
        - id: The unique identifier for the observation.
        - ra: The right ascension of the observation in degrees.
        - dec: The declination of the observation in degrees.
        - magnitude: The magnitude of the observation.
        - mjd: The modified Julian date of the observation.
        - band: The band of the observation.
        - obscode: The observatory code for the observation.
    """
    zp = np.mean(image_collection["zeroPoint"])

    ids = []
    ras = []
    decs = []
    mags = []
    mjds = []
    bands = []
    obscodes = []

    all_times = results.table.meta["mjd_mid"]
    all_bands = image_collection["band"]

    _, unique_indices = get_unique_obstimes(image_collection["mjd_mid"])
    first_of_each_frame = np.array([i[0] for i in unique_indices])

    for i, row in enumerate(results):
        if "obs_valid" in results.table.colnames:
            valid_obs = row["obs_valid"]
        else:
            valid_obs = np.full(row["obs_count"], True)
        num_valid = row["obs_count"]

        # need to figure out a better way to do this
        if batch_id is not None:
            ids.append([f"{batch_id}-{i}-{j}" for j in range(num_valid)])
        else:
            ids.append([f"{i}-{j}" for j in range(num_valid)])

        ras.append(row["img_ra"][valid_obs])
        decs.append(row["img_dec"][valid_obs])

        mags.append([get_magnitude(row["flux"], zp)] * num_valid)
        mjds.append(all_times[valid_obs])
        bands.append(all_bands[first_of_each_frame][valid_obs])
        obscodes.append([obscode] * num_valid)

    final_df = pd.DataFrame()
    final_df["id"] = np.concatenate(ids)
    final_df["ra"] = np.concatenate(ras)
    final_df["dec"] = np.concatenate(decs)
    final_df["magnitude"] = np.concatenate(mags)
    final_df["mjd"] = np.concatenate(mjds)
    final_df["band"] = np.concatenate(bands)
    final_df["obscode"] = np.concatenate(obscodes)

    return final_df
