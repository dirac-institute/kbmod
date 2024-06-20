"""Functions to do basic filtering of Results, such as filtering the points based
on likleihood or the rows based on time range.
"""

import astropy.time
import logging
import numpy as np

from kbmod.results import Results

logger = logging.getLogger(__name__)


def apply_likelihood_clipping(result_data, lower_bnd=-np.inf, upper_bnd=np.inf):
    """Filter individual time steps with points above or below given likelihood
    thresholds. Applies the filtering to the result_data in place.

    Parameters
    ----------
    result_data : `Results`
        The values from trajectories. This data gets modified directly by the filtering.
    lower_bnd : `float`
        The minimum likleihood for a single observation. Default = -inf.
    upper_bnd : `float`
        The maximum likelihood for a single observation. Default = inf.
    """
    logger.info(f"Threshold clipping {len(result_data)} results with " f"bounds=[{lower_bnd}, {upper_bnd}]")

    lh = result_data.compute_likelihood_curves(filter_obs=True, mask_value=np.nan)
    obs_valid = np.isfinite(lh) & (lh <= upper_bnd) & (lh >= lower_bnd)
    result_data.update_obs_valid(obs_valid)


def apply_time_range_filter(result_data, mjds, min_days=np.inf, colname=None):
    """Filter any row that does not valid observations that cover at least
    ``threshold`` days. Applies the filtering to the result_data in place.

    Parameters
    ----------
    result_data : `Results`
        The values from trajectories. This data gets modified directly by the filtering.
    mjds : `numpy.ndarray`
        An array of the timestamps for each observation time.
    min_days : `float`
        The minimum time length from the first to last valid observation (in days).
        Default = inf
    colname : `str`
        If provided, adds the duration as a column to results.
    """
    num_rows = len(result_data)
    logger.info(f"Filtering {num_rows} results on direction >= {min_days}")
    if num_rows == 0:
        return

    # Create a masked time array to use for the min/max.
    time_mat = np.repeat(np.array([mjds], dtype=float), num_rows, axis=0)
    time_mat = result_data.mask_based_on_invalid_obs(time_mat, np.nan)

    # Compute the duration of each result.
    delta = np.nanmax(time_mat, axis=1) - np.nanmin(time_mat, axis=1)
    if colname is not None:
        result_data.table[colname] = delta
    result_data.filter_rows(delta >= min_days, f"duration>={min_days}")


def apply_unique_day_filter(result_data, mjds, min_days, min_per_day=1, colname=None):
    """Filter any row that does not valid observations that occur on at least
    ``min_days`` unique days. Applies the filtering to the result_data in place.

    Based on Wilson's code from the two day analysis notebook.

    Parameters
    ----------
    result_data : `Results`
        The values from trajectories. This data gets modified directly by the filtering.
    mjds : `numpy.ndarray`
        An array of the timestamps for each observation time.
    min_days : `int`
        The minimum number of days on which we need a valid observation.
    min_per_day : `int`
        The minimum number of observations we need to see on a single day in order
        to count that day. Default = 1
    colname : `str`
        If provided, adds the duration as a column to results.
        Default = None
    """
    num_rows = len(result_data)
    logger.info(f"Filtering {num_rows} results on unique number of days >= {min_days}")
    if num_rows == 0:
        return

    # Transform the list of MJDs into a list of strings with the calendar date.
    dates_info = astropy.time.Time(mjds, format="mjd").to_value("datetime")
    date_strs = [f"{date.year}_{date.month}_{date.day}" for date in dates_info]

    # Create a masked matrix of dates where invalid observations have a data of ''.
    date_mat = np.repeat(np.array([date_strs]), num_rows, axis=0)
    date_mat = result_data.mask_based_on_invalid_obs(date_mat, "")

    # Count the number of unique days long each row.
    def _count_unique_days(row, min_per_day):
        vals, counts = np.unique(row, return_counts=True)
        counts[vals == ""] = 0  # Zero out masked values
        return len(np.argwhere(counts >= min_per_day))

    unique_days = np.apply_along_axis(_count_unique_days, 1, date_mat, min_per_day=min_per_day)
    if colname is not None:
        result_data.table[colname] = unique_days
    result_data.filter_rows(unique_days >= min_days, f"unique_days>={min_days}")
