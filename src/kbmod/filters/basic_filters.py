"""Functions to do basic filtering of Results, such as filtering the points based
on likleihood or the rows based on time range.
"""

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
