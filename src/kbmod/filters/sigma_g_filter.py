"""Functions to help with the SigmaG clipping.

For more details see:
Sifting Through the Static: Moving Object Detection in Difference Images
by Smotherman et. al. 2021
"""

import logging
import numpy as np
import os
import torch

from scipy.special import erfinv

from kbmod.results import Results
from kbmod.search import DebugTimer


logger = logging.getLogger(__name__)


class SigmaGClipping:
    """This class contains the basic information for performing SigmaG clipping.

    Attributes
    ----------
    low_bnd : `float`
        The lower bound of the interval to use to estimate the standard deviation.
    high_bnd : `float`
        The upper bound of the interval to use to estimate the standard deviation.
    n_sigma : `float`
        The number of standard deviations to use for the bound.
    clip_negative : `bool`
        A Boolean indicating whether to use negative values when computing
        standard deviation.
    coeff : `float`
        The precomputed coefficient based on the given bounds.
    """

    def __init__(self, low_bnd=25, high_bnd=75, n_sigma=2, clip_negative=False):
        if low_bnd > high_bnd or low_bnd <= 0.0 or high_bnd >= 100.0:
            raise ValueError(f"Invalid bounds [{low_bnd}, {high_bnd}]")
        if n_sigma <= 0.0:
            raise ValueError(f"Invalid n_sigma {n_sigma}")

        self.low_bnd = low_bnd
        self.high_bnd = high_bnd
        self.n_sigma = n_sigma
        self.coeff = SigmaGClipping.find_sigma_g_coeff(low_bnd, high_bnd)
        self.clip_negative = clip_negative

    @staticmethod
    def find_sigma_g_coeff(low_bnd, high_bnd):
        """Compute the sigma G coefficient from the upper and lower bounds
        of the percentiles.

        Parameters
        ----------
        low_bnd : `float`
            The lower bound of the percentile on a scale of [0, 100].
        high_bnd : `float`
            The lower bound of the percentile on a scale of [0, 100].

        Returns
        -------
        result : `float`
            The corresponding sigma G coefficient.

        Raises
        ------
        Raises a ``ValueError`` is the bounds are invalid.
        """
        if (high_bnd <= low_bnd) or (low_bnd < 0) or (high_bnd > 100):
            raise ValueError(f"Invalid percentiles for sigma G coefficient [{low_bnd}, {high_bnd}]")
        x1 = SigmaGClipping.invert_gauss_cdf(low_bnd / 100.0)
        x2 = SigmaGClipping.invert_gauss_cdf(high_bnd / 100.0)
        return 1 / (x2 - x1)

    @staticmethod
    def invert_gauss_cdf(z):
        if z < 0.5:
            sign = -1
        else:
            sign = 1
        x = sign * np.sqrt(2) * erfinv(sign * (2 * z - 1))
        return float(x)

    def compute_clipped_sigma_g(self, lh):
        """Compute the SigmaG clipping on the given likelihood curve.
        Points are eliminated if they are more than n_sigma*sigmaG away from the median.

        Parameters
        ----------
        lh : numpy array
            A single likelihood curve.

        Returns
        -------
        good_index: numpy array
            The indices that pass the filtering for a given set of curves.
        """
        if self.clip_negative:
            # Skip entries where we will clip everything (all lh < 0).
            if np.count_nonzero(lh > 0) == 0:
                return np.array([])
            lower_per, median, upper_per = np.percentile(lh[lh > 0], [self.low_bnd, 50, self.high_bnd])
        else:
            lower_per, median, upper_per = np.percentile(lh, [self.low_bnd, 50, self.high_bnd])

        delta = max(upper_per - lower_per, 1e-8)
        sigmaG = self.coeff * delta
        nSigmaG = self.n_sigma * sigmaG

        good_index = np.where(np.logical_and(lh > median - nSigmaG, lh < median + nSigmaG))[0]
        return good_index

    def compute_clipped_sigma_g_matrix(self, lh):
        """Compute the SigmaG clipping on a matrix containing curves where
        each row is a single curve at different time points.
        Points are eliminated if they are more than n_sigma*sigmaG away from the median.

        Parameters
        ----------
        lh : `numpy.ndarray`
            A N x T matrix with N curves, each with T time steps.

        Returns
        -------
        index_valid : `numpy.ndarray`
            A N x T matrix of Booleans indicating if each point is valid (True)
            or has been filtered (False).
        """
        # Use a GPU if one is available.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Move the likelihood data to the device.
        torch_lh = torch.tensor(lh, device=device, dtype=torch.float64)

        # Mask out the negative values (if clip negative is true).
        if self.clip_negative:
            masked_lh = torch.where(torch_lh > 0.0, torch_lh, np.nan)
        else:
            masked_lh = torch_lh

        # Compute the quantiles for each row.
        q_values = torch.tensor(
            [self.low_bnd / 100.0, 0.5, self.high_bnd / 100.0],
            device=device,
            dtype=torch.float64,
        )
        lower_per, median, upper_per = torch.nanquantile(masked_lh, q_values, dim=1)

        # Compute the bounds for each row, enforcing a minimum gap in case all the
        # points are identical (upper_per == lower_per).
        delta = upper_per - lower_per
        delta[delta < 1e-8] = 1e-8
        nSigmaG = self.n_sigma * self.coeff * delta

        num_rows = lh.shape[0]
        num_cols = lh.shape[1]
        lower_bnd = torch.reshape((median - nSigmaG), (num_rows, 1)).expand(-1, num_cols)
        upper_bnd = torch.reshape((median + nSigmaG), (num_rows, 1)).expand(-1, num_cols)

        # Check whether the values fall within the bounds.
        index_valid = torch.isfinite(torch_lh) & (torch_lh < upper_bnd) & (torch_lh > lower_bnd)

        # Return as a numpy array on the CPU.
        return index_valid.cpu().numpy().astype(bool)


def apply_clipped_sigma_g(clipper, result_data):
    """This function applies a clipped median filter to the results of a KBMOD
    search using sigmaG as a robust estimater of standard deviation.

    Parameters
    ----------
    clipper : `SigmaGClipping`
        The object to apply the SigmaG clipping.
    result_data : `Results`
        The values from trajectories. This data gets modified directly by the filtering.
    """
    if len(result_data) == 0:
        logger.info("SigmaG Clipping : skipping, nothing to filter.")
        return

    filter_timer = DebugTimer("sigma-g filtering", logger)
    lh = result_data.compute_likelihood_curves(filter_obs=True, mask_value=np.nan)
    obs_valid = clipper.compute_clipped_sigma_g_matrix(lh)
    result_data.update_obs_valid(obs_valid)
    filter_timer.stop()
