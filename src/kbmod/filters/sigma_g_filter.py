"""Functions to help with the SigmaG clipping.

For more details see:
Sifting Through the Static: Moving Objectg Detection in Difference Images
by Smotherman et. al. 2021
"""

import multiprocessing as mp
import numpy as np
from scipy.special import erfinv

from kbmod.result_list import ResultList, ResultRow
from kbmod.results import Results


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
        self.clip_negative = clip_negative
        self.n_sigma = n_sigma
        self.coeff = SigmaGClipping.find_sigma_g_coeff(low_bnd, high_bnd)

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

        # Its unclear why we only filter zeros for one of the two cases, but leaving the logic in
        # to stay consistent with the original code.
        if self.clip_negative:
            good_index = np.where(
                np.logical_and(lh != 0, np.logical_and(lh > median - nSigmaG, lh < median + nSigmaG))
            )[0]
        else:
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
        if self.clip_negative:
            # We mask out the values less than zero so they are not used in the median computation.
            masked_lh = np.copy(lh)
            masked_lh[lh <= 0] = np.NAN
            lower_per, median, upper_per = np.nanpercentile(
                masked_lh, [self.low_bnd, 50, self.high_bnd], axis=1
            )
        else:
            lower_per, median, upper_per = np.nanpercentile(lh, [self.low_bnd, 50, self.high_bnd], axis=1)

        # Compute the bounds for each row, enforcing a minimum gap in case all the
        # points are identical (upper_per == lower_per).
        delta = upper_per - lower_per
        delta[delta < 1e-8] = 1e-8
        nSigmaG = self.n_sigma * self.coeff * delta

        num_cols = lh.shape[1]
        lower_bnd = np.repeat(np.array([median - nSigmaG]).T, num_cols, axis=1)
        upper_bnd = np.repeat(np.array([median + nSigmaG]).T, num_cols, axis=1)

        # Its unclear why we only filter zeros for one of the two cases, but leaving the logic in
        # to stay consistent with the original code.
        if self.clip_negative:
            index_valid = np.isfinite(lh) & (lh != 0) & (lh < upper_bnd) & (lh > lower_bnd)
        else:
            index_valid = np.isfinite(lh) & (lh < upper_bnd) & (lh > lower_bnd)
        return index_valid


def apply_single_clipped_sigma_g(clipper, result):
    """This function applies a clipped median filter to a single result from
    KBMOD using sigmaG as a robust estimater of standard deviation.

    Parameters
    ----------
    clipper : `SigmaGClipping`
        The object to apply the SigmaG clipping.
    result : `ResultRow`
        The result details. This data gets modified directly by the filtering.
    """
    single_res = clipper.compute_clipped_sigma_g(result.likelihood_curve)
    result.filter_indices(single_res)


def apply_clipped_sigma_g(clipper, result_data, num_threads=1):
    """This function applies a clipped median filter to the results of a KBMOD
    search using sigmaG as a robust estimater of standard deviation.

    Parameters
    ----------
    clipper : `SigmaGClipping`
        The object to apply the SigmaG clipping.
    result_data : `ResultList` or `Results`
        The values from trajectories. This data gets modified directly by the filtering.
    num_threads : `int`
        The number of threads to use.
    """
    if type(result_data) is Results:
        lh = result_data.compute_likelihood_curves(filter_obs=True, mask_value=np.NAN)
        obs_valid = clipper.compute_clipped_sigma_g_matrix(lh)
        result_data.update_obs_valid(obs_valid)
        return

    # TODO: Remove this logic once we have switched over to Results.
    if num_threads > 1:
        lh_list = [[row.likelihood_curve] for row in result_data.results]

        keep_idx_results = []
        pool = mp.Pool(processes=num_threads)
        keep_idx_results = pool.starmap_async(clipper.compute_clipped_sigma_g, lh_list)
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()

        for i, res in enumerate(keep_idx_results):
            result_data.results[i].filter_indices(res)
    else:
        for row in result_data.results:
            apply_single_clipped_sigma_g(clipper, row)
