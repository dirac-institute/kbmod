"""Functions to help with the SigmaG clipping.

For more details see:
Sifting Through the Static: Moving Objectg Detection in Difference Images
by Smotherman et. al. 2021
"""

from functools import partial
from jax import jit, vmap
import jax.numpy as jnp
import logging
import numpy as np
import os
from scipy.special import erfinv

from kbmod.results import Results
from kbmod.search import DebugTimer


logger = logging.getLogger(__name__)


def sigma_g_jax(data, low_bnd, high_bnd, n_sigma, coeff, clip_negative):
    """The core function for performing a sigma filtering on a series of data points
    with clipped_negative. These are typically likelihoods for KBMOD.

    Parameters
    ----------
    data : `numpy.ndarray`
        A length T matrix of data points for filtering.
    low_bnd : `float`
        The lower bound of the interval to use to estimate the standard deviation.
    high_bnd : `float`
        The upper bound of the interval to use to estimate the standard deviation.
    n_sigma : `float`
        The number of standard deviations to use for the bound.
    coeff : `float`
        The precomputed coefficient based on the given bounds.
    clip_negative : `bool`
        A Boolean indicating whether to use negative values when computing
        standard deviation.

    Returns
    -------
    index_valid : `numpy.ndarray`
        A length T array of Booleans indicating if each point is valid (True)
        or has been filtered (False).
    """
    # Compute the percentiles for this array of values. If we are clipping the negatives then only
    # use the positive points.
    masked_data = jnp.where((not clip_negative) | (data > 0.0), data, jnp.nan)
    lower_per, median, upper_per = jnp.nanpercentile(masked_data, jnp.array([low_bnd, 50, high_bnd]))

    # Compute the bounds for each row, enforcing a minimum gap in case all the
    # points are identical (upper_per == lower_per).
    delta = upper_per - lower_per
    nSigmaG = n_sigma * coeff * jnp.where(delta > 1e-8, delta, 1e-8)

    index_valid = jnp.isfinite(data) & (data <= median + nSigmaG) & (data >= median - nSigmaG)
    return index_valid


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

        # Create compiled vmapped functions that applies the Sigma G filtering
        # with the given parameters.
        base_fn = partial(
            sigma_g_jax,
            low_bnd=self.low_bnd,
            high_bnd=self.high_bnd,
            n_sigma=self.n_sigma,
            coeff=self.coeff,
            clip_negative=self.clip_negative,
        )
        self.sigma_g_jax_fn = vmap(jit(base_fn))

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
        # We need to
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        jnp_lh = jnp.array(lh)
        jpn_inds_valid = self.sigma_g_jax_fn(jnp_lh).block_until_ready()

        # Make sure the inds_valid is on CPU as a numpy array.
        inds_valid = np.array(jpn_inds_valid)
        return inds_valid


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
