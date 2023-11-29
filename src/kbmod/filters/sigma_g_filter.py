"""Functions to help with the SigmaG clipping.

For more details see:
Sifting Through the Static: Moving Objectg Detection in Difference Images
by Smotherman et. al. 2021
"""

import multiprocessing as mp
import numpy as np
from scipy.special import erfinv

from kbmod.result_list import ResultList, ResultRow


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


def apply_clipped_sigma_g(params, result_list, num_threads=1):
    """This function applies a clipped median filter to the results of a KBMOD
    search using sigmaG as a robust estimater of standard deviation.

    Parameters
    ----------
    params : `SigmaGClipping`
        The object to apply the SigmaG clipping.
    result_list : `ResultList`
        The values from trajectories. This data gets modified directly by the filtering.
    num_threads : `int`
        The number of threads to use.
    """
    if num_threads > 1:
        lh_list = [[row.likelihood_curve] for row in result_list.results]

        keep_idx_results = []
        pool = mp.Pool(processes=num_threads)
        keep_idx_results = pool.starmap_async(params.compute_clipped_sigma_g, lh_list)
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()

        for i, res in enumerate(keep_idx_results):
            result_list.results[i].filter_indices(res)
    else:
        for i, row in enumerate(result_list.results):
            single_res = params.compute_clipped_sigma_g(row.likelihood_curve)
            row.filter_indices(single_res)
