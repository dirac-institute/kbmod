import multiprocessing as mp
import os
import time

import numpy as np
from scipy.special import erfinv

import kbmod.search as kb

from .file_utils import *
from .filters.clustering_filters import DBSCANFilter
from .filters.stats_filters import CombinedStatsFilter
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .result_list import ResultList, ResultRow


logger = kb.Logging.getLogger(__name__)


class PostProcess:
    """This class manages the post-processing utilities used to filter out and
    otherwise remove false positives from the KBMOD search. This includes,
    for example, kalman filtering to remove outliers, stamp filtering to remove
    results with non-Gaussian postage stamps, and clustering to remove similar
    results.
    """

    def __init__(self, config, mjds):
        self.num_cores = config["num_cores"]
        self.num_obs = config["num_obs"]
        self.sigmaG_lims = config["sigmaG_lims"]
        self.eps = config["eps"]
        self.cluster_type = config["cluster_type"]
        self.cluster_function = config["cluster_function"]
        self.clip_negative = config["clip_negative"]
        self._mjds = mjds

    def load_and_filter_results(
        self,
        search,
        lh_level,
        chunk_size=500000,
        max_lh=1e9,
    ):
        """This function loads results that are output by the gpu grid search.
        Results are loaded in chunks and evaluated to see if the minimum
        likelihood level has been reached. If not, another chunk of results is
        fetched. The results are filtered using a clipped-sigmaG filter as they
        are loaded and only the passing results are kept.

        Parameters
        ----------
        search : `kbmod.search`
            The search function object.
        lh_level : float
            The minimum likelihood theshold for an acceptable result. Results below
            this likelihood level will be discarded.
        chunk_size : int
            The number of results to load at a given time from search.
        max_lh : float
            The maximum likelihood threshold for an acceptable results.
            Results ABOVE this likelihood level will be discarded.

        Returns
        -------
        keep : `ResultList`
            A ResultList object containing values from trajectories.
        """
        keep = ResultList(self._mjds)
        likelihood_limit = False
        res_num = 0
        total_count = 0

        # Set up the clipped sigmaG filter.
        if self.sigmaG_lims is not None:
            bnds = self.sigmaG_lims
        else:
            bnds = [25, 75]
        clipper = SigmaGClipping(bnds[0], bnds[1], 2, self.clip_negative)

        # Set up the combined stats filter.
        if lh_level > 0.0:
            stats_filter = CombinedStatsFilter(min_obs=self.num_obs, min_lh=lh_level)
        else:
            stats_filter = CombinedStatsFilter(min_obs=self.num_obs)

        logger.info("Retrieving Results")
        while likelihood_limit is False:
            logger.info("Getting results...")
            results = search.get_results(res_num, chunk_size)
            logger.info("Chunk Start = %i" % res_num)
            logger.info("Chunk Max Likelihood = %.2f" % results[0].lh)
            logger.info("Chunk Min. Likelihood = %.2f" % results[-1].lh)

            result_batch = ResultList(self._mjds)
            for i, trj in enumerate(results):
                # Stop as soon as we hit a result below our limit, because anything after
                # that is not guarrenteed to be valid due to potential on-GPU filtering.
                if trj.lh < lh_level:
                    likelihood_limit = True
                    break

                if trj.lh < max_lh:
                    row = ResultRow(trj, len(self._mjds))
                    psi_curve = np.array(search.get_psi_curves(trj))
                    phi_curve = np.array(search.get_phi_curves(trj))
                    row.set_psi_phi(psi_curve, phi_curve)
                    result_batch.append_result(row)
                    total_count += 1

            batch_size = result_batch.num_results()
            logger.info("Extracted batch of %i results for total of %i" % (batch_size, total_count))
            if batch_size > 0:
                apply_clipped_sigma_g(clipper, result_batch, self.num_cores)
                result_batch.apply_filter(stats_filter)

                # Add the results to the final set.
                keep.extend(result_batch)
            res_num += chunk_size
        return keep

    def apply_clustering(self, result_list, cluster_params):
        """This function clusters results that have similar trajectories.

        Parameters
        ----------
        result_list : `ResultList`
            The values from trajectories. This data gets modified directly by
            the filtering.
        cluster_params : dict
            Contains values concerning the image and search settings including:
            x_size, y_size, vel_lims, ang_lims, and mjd.
        """
        # Skip clustering if there is nothing to cluster.
        if result_list.num_results() == 0:
            return
        logger.info("Clustering %i results" % result_list.num_results())

        # Do the clustering and the filtering.
        f = DBSCANFilter(
            self.cluster_type,
            self.eps,
            cluster_params["x_size"],
            cluster_params["y_size"],
            cluster_params["vel_lims"],
            cluster_params["ang_lims"],
            cluster_params["mjd"],
        )
        result_list.apply_batch_filter(f)
