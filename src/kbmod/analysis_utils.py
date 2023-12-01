import multiprocessing as mp
import os
import time

import numpy as np
from scipy.special import erfinv

import kbmod.search as kb

from .file_utils import *
from .filters.clustering_filters import DBSCANFilter
from .filters.stats_filters import LHFilter, NumObsFilter
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .result_list import ResultList, ResultRow


class PostProcess:
    """This class manages the post-processing utilities used to filter out and
    otherwise remove false positives from the KBMOD search. This includes,
    for example, kalman filtering to remove outliers, stamp filtering to remove
    results with non-Gaussian postage stamps, and clustering to remove similar
    results.
    """

    def __init__(self, config, mjds):
        self.coeff = None
        self.num_cores = config["num_cores"]
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

        print("---------------------------------------")
        print("Retrieving Results")
        print("---------------------------------------")
        while likelihood_limit is False:
            print("Getting results...")
            results = search.get_results(res_num, chunk_size)
            print("---------------------------------------")
            print("Chunk Start = %i" % res_num)
            print("Chunk Max Likelihood = %.2f" % results[0].lh)
            print("Chunk Min. Likelihood = %.2f" % results[-1].lh)
            print("---------------------------------------")

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
            print("Extracted batch of %i results for total of %i" % (batch_size, total_count))
            if batch_size > 0:
                apply_clipped_sigma_g(clipper, result_batch, self.num_cores)
                result_batch.apply_filter(NumObsFilter(3))

                # Apply the likelihood filter if one is provided.
                if lh_level > 0.0:
                    result_batch.apply_filter(LHFilter(lh_level, None))

                # Add the results to the final set.
                keep.extend(result_batch)
            res_num += chunk_size
        return keep

    def get_all_stamps(self, result_list, search, stamp_radius):
        """Get the stamps for the final results from a kbmod search.

        Parameters
        ----------
        result_list : `ResultList`
            The values from trajectories. The stamps are inserted into this data structure.
        search : `kbmod.StackSearch`
            The search object
        stamp_radius : int
            The radius of the stamps to create.
        """
        stamp_edge = stamp_radius * 2 + 1
        for row in result_list.results:
            stamps = kb.StampCreator.get_stamps(search.get_imagestack(), row.trajectory, stamp_radius)
            # TODO: a way to avoid a copy here would be to do
            # np.array([s.image for s in stamps], dtype=np.single, copy=False)
            # but that could cause a problem with reference counting at the m
            # moment. The real fix is to make the stamps return Image not
            # RawImage, return the Image and avoid a reference to a private
            # attribute. This risks collecting RawImage but leaving a dangling
            # ref to its private field. That's a fix for another time.
            row.all_stamps = np.array([stamp.image for stamp in stamps])

    def apply_stamp_filter(
        self,
        result_list,
        search,
        center_thresh=0.03,
        peak_offset=[2.0, 2.0],
        mom_lims=[35.5, 35.5, 1.0, 0.25, 0.25],
        chunk_size=1000000,
        stamp_type="sum",
        stamp_radius=10,
    ):
        """This function filters result postage stamps based on their Gaussian
        Moments. Results with stamps that are similar to a Gaussian are kept.

        Parameters
        ----------
        result_list : `ResultList`
            The values from trajectories. This data gets modified directly by
            the filtering.
        search : `kbmod.StackSearch`
            The search object.
        center_thresh : float
            The fraction of the total flux that must be contained in a single
            central pixel.
        peak_offset : list of floats
            How far the brightest pixel in the stamp can be from the central
            pixel.
        mom_lims : list of floats
            The maximum limit of the xx, yy, xy, x, and y central moments of
            the stamp.
        chunk_size : int
            How many stamps to load and filter at a time.
        stamp_type : string
            Which method to use to generate stamps.
            One of 'median', 'cpp_median', 'mean', 'cpp_mean', or 'sum'.
        stamp_radius : int
            The radius of the stamp.
        """
        # Set the stamp creation and filtering parameters.
        params = kb.StampParameters()
        params.radius = stamp_radius
        params.do_filtering = True
        params.center_thresh = center_thresh
        params.peak_offset_x = peak_offset[0]
        params.peak_offset_y = peak_offset[1]
        params.m20_limit = mom_lims[0]
        params.m02_limit = mom_lims[1]
        params.m11_limit = mom_lims[2]
        params.m10_limit = mom_lims[3]
        params.m01_limit = mom_lims[4]

        if stamp_type == "cpp_median" or stamp_type == "median":
            params.stamp_type = kb.StampType.STAMP_MEDIAN
        elif stamp_type == "cpp_mean" or stamp_type == "mean":
            params.stamp_type = kb.StampType.STAMP_MEAN
        else:
            params.stamp_type = kb.StampType.STAMP_SUM

        # Save some useful helper data.
        num_times = search.get_num_images()
        all_valid_inds = []

        # Run the stamp creation and filtering in batches of chunk_size.
        print("---------------------------------------")
        print("Applying Stamp Filtering")
        print("---------------------------------------", flush=True)
        start_time = time.time()
        start_idx = 0
        if result_list.num_results() <= 0:
            print("Skipping. Nothing to filter.")
            return

        print("Stamp filtering %i results" % result_list.num_results())
        while start_idx < result_list.num_results():
            end_idx = min([start_idx + chunk_size, result_list.num_results()])

            # Create a subslice of the results and the Boolean indices.
            # Note that the sum stamp type does not filter out lc_index.
            inds_to_use = [i for i in range(start_idx, end_idx)]
            trj_slice = [result_list.results[i].trajectory for i in inds_to_use]
            if params.stamp_type != kb.StampType.STAMP_SUM:
                bool_slice = [result_list.results[i].valid_indices_as_booleans() for i in inds_to_use]
            else:
                # For the sum stamp, use all the indices for each trajectory.
                all_true = [True] * num_times
                bool_slice = [all_true for _ in inds_to_use]

            # Create and filter the results, using the GPU if there is one and enough
            # trajectories to make it worthwhile.
            stamps_slice = kb.StampCreator.get_coadded_stamps(
                search.get_imagestack(),
                trj_slice,
                bool_slice,
                params,
                kb.HAS_GPU and len(trj_slice) > 100,
            )
            # TODO: a way to avoid a copy here would be to do
            # np.array([s.image for s in stamps], dtype=np.single, copy=False)
            # but that could cause a problem with reference counting at the m
            # moment. The real fix is to make the stamps return Image not
            # RawImage and avoid reference to an private attribute and risking
            # collecting RawImage but leaving a dangling ref to the attribute.
            # That's a fix for another time so I'm leaving it as a copy here
            for ind, stamp in enumerate(stamps_slice):
                if stamp.width > 1:
                    result_list.results[ind + start_idx].stamp = np.array(stamp.image)
                    all_valid_inds.append(ind + start_idx)

            # Move to the next chunk.
            start_idx += chunk_size

        # Do the actual filtering of results
        result_list.filter_results(all_valid_inds)
        print("Keeping %i results" % result_list.num_results(), flush=True)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("{:.2f}s elapsed".format(time_elapsed))

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
        print("Clustering %i results" % result_list.num_results(), flush=True)

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
