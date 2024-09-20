import os
import time

import numpy as np

import kbmod.search as kb

from .configuration import SearchConfiguration
from .filters.clustering_filters import apply_clustering
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .filters.stamp_filters import append_all_stamps, append_coadds, get_coadds_and_filter_results

from .results import Results
from .trajectory_generator import create_trajectory_generator
from .work_unit import WorkUnit


logger = kb.Logging.getLogger(__name__)


class SearchRunner:
    """A class to run the KBMOD grid search."""

    def __init__(self):
        pass

    def load_and_filter_results(self, search, config):
        """This function loads results that are output by the gpu grid search.
        Results are loaded in chunks and evaluated to see if the minimum
        likelihood level has been reached. If not, another chunk of results is
        fetched. The results are filtered using a clipped-sigmaG filter as they
        are loaded and only the passing results are kept.

        Parameters
        ----------
        search : `kbmod.search`
            The search function object.
        config : `SearchConfiguration`
            The configuration parameters
        chunk_size : int
            The number of results to load at a given time from search.

        Returns
        -------
        keep : `Results`
            A Results object containing values from trajectories.
        """
        # Parse and check the configuration parameters.
        num_obs = config["num_obs"]
        sigmaG_lims = config["sigmaG_lims"]
        clip_negative = config["clip_negative"]
        lh_level = config["lh_level"]
        max_lh = config["max_lh"]
        chunk_size = config["chunk_size"]
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk size {chunk_size}")

        # Set up the list of results.
        do_tracking = config["track_filtered"]
        img_stack = search.get_imagestack()
        num_times = img_stack.img_count()
        keep = Results(track_filtered=do_tracking)

        # Set up the clipped sigmaG filter.
        if sigmaG_lims is not None:
            bnds = sigmaG_lims
        else:
            bnds = [25, 75]
        clipper = SigmaGClipping(bnds[0], bnds[1], 2, clip_negative)

        total_found = search.get_number_total_results()
        logger.info(f"Retrieving Results (total={total_found})")
        likelihood_limit = False
        res_num = 0
        total_count = 0

        # Keep retrieving results until they fall below the threshold or we run out of results.
        while likelihood_limit is False and res_num < total_found:
            logger.info(f"Chunk Start = {res_num} (size={chunk_size})")
            results = search.get_results(res_num, chunk_size)
            logger.info(f"Chunk Max Likelihood = {results[0].lh}")
            logger.info(f"Chunk Min. Likelihood = {results[-1].lh}")

            trj_batch = []
            for i, trj in enumerate(results):
                # Stop as soon as we hit a result below our limit, because anything after
                # that is not guarrenteed to be valid due to potential on-GPU filtering.
                if trj.lh < lh_level:
                    likelihood_limit = True
                    break

                if trj.lh < max_lh:
                    trj_batch.append(trj)
                    total_count += 1

            batch_size = len(trj_batch)
            logger.info(f"Extracted batch of {batch_size} results for total of {total_count}")

            if batch_size > 0:
                psi_batch = search.get_psi_curves(trj_batch)
                phi_batch = search.get_phi_curves(trj_batch)

                result_batch = Results.from_trajectories(trj_batch, track_filtered=do_tracking)
                result_batch.add_psi_phi_data(psi_batch, phi_batch)

                # Do the sigma-G filtering and subsequent stats filtering.
                apply_clipped_sigma_g(clipper, result_batch)
                obs_row_mask = result_batch["obs_count"] >= num_obs
                result_batch.filter_rows(obs_row_mask, "obs_count")
                logger.debug(f"After obs_count >= {num_obs}. Batch size = {len(result_batch)}")

                if lh_level > 0.0:
                    lh_row_mask = result_batch["likelihood"] >= lh_level
                    result_batch.filter_rows(lh_row_mask, "likelihood")
                    logger.debug(f"After likelihood >= {lh_level}. Batch size = {len(result_batch)}")

                # Add the results to the final set.
                keep.extend(result_batch)
            res_num += chunk_size
        return keep

    def do_gpu_search(self, config, stack, trj_generator):
        """Performs search on the GPU.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        stack : `ImageStack`
            The stack before the masks have been applied. Modified in-place.
        trj_generator : `TrajectoryGenerator`
            The object to generate the candidate trajectories for each pixel.

        Returns
        -------
        keep : `Results`
            The results.
        """
        # Create the search object which will hold intermediate data and results.
        search = kb.StackSearch(stack)

        width = search.get_image_width()
        height = search.get_image_height()

        # Set the search bounds.
        if config["x_pixel_bounds"] and len(config["x_pixel_bounds"]) == 2:
            search.set_start_bounds_x(config["x_pixel_bounds"][0], config["x_pixel_bounds"][1])
        elif config["x_pixel_buffer"] and config["x_pixel_buffer"] > 0:
            search.set_start_bounds_x(-config["x_pixel_buffer"], width + config["x_pixel_buffer"])

        if config["y_pixel_bounds"] and len(config["y_pixel_bounds"]) == 2:
            search.set_start_bounds_y(config["y_pixel_bounds"][0], config["y_pixel_bounds"][1])
        elif config["y_pixel_buffer"] and config["y_pixel_buffer"] > 0:
            search.set_start_bounds_y(-config["y_pixel_buffer"], height + config["y_pixel_buffer"])

        # Set the results per pixel.
        search.set_results_per_pixel(config["results_per_pixel"])

        search_timer = kb.DebugTimer("grid search", logger)
        logger.debug(f"{trj_generator}")

        # If we are using gpu_filtering, enable it and set the parameters.
        if config["gpu_filter"]:
            logger.debug("Using in-line GPU sigmaG filtering methods")
            coeff = SigmaGClipping.find_sigma_g_coeff(
                config["sigmaG_lims"][0],
                config["sigmaG_lims"][1],
            )
            search.enable_gpu_sigmag_filter(
                np.array(config["sigmaG_lims"]) / 100.0,
                coeff,
                config["lh_level"],
            )

        # If we are using an encoded image representation on GPU, enable it and
        # set the parameters.
        if config["encode_num_bytes"] > 0:
            search.enable_gpu_encoding(config["encode_num_bytes"])

        # Do the actual search.
        candidates = [trj for trj in trj_generator]
        search.search_all(candidates, int(config["num_obs"]))
        search_timer.stop()

        # Load the results.
        keep = self.load_and_filter_results(search, config)
        return keep

    def run_search(self, config, stack, trj_generator=None):
        """This function serves as the highest-level python interface for starting
        a KBMOD search given an ImageStack and SearchConfiguration.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        stack : `ImageStack`
            The stack before the masks have been applied. Modified in-place.
        trj_generator : `TrajectoryGenerator`, optional
            The object to generate the candidate trajectories for each pixel.
            If None uses the default EclipticCenteredSearch

        Returns
        -------
        keep : `Results`
            The results.
        """
        if not kb.HAS_GPU:
            logger.warning("Code was compiled without GPU.")

        full_timer = kb.DebugTimer("KBMOD", logger)

        # Apply the mask to the images.
        if config["do_mask"]:
            for i in range(stack.img_count()):
                stack.get_single_image(i).apply_mask(0xFFFFFF)

        # Perform the actual search.
        if trj_generator is None:
            trj_generator = create_trajectory_generator(config, work_unit=None)
        keep = self.do_gpu_search(config, stack, trj_generator)

        if config["do_stamp_filter"]:
            stack.copy_to_gpu()
            get_coadds_and_filter_results(keep, stack, config)
            stack.clear_from_gpu()

        if config["do_clustering"]:
            cluster_timer = kb.DebugTimer("clustering", logger)
            mjds = [stack.get_obstime(t) for t in range(stack.img_count())]
            cluster_params = {
                "cluster_type": config["cluster_type"],
                "cluster_eps": config["cluster_eps"],
                "cluster_v_scale": config["cluster_v_scale"],
                "times": np.array(mjds),
            }
            apply_clustering(keep, cluster_params)
            cluster_timer.stop()

        # Generate additional coadded stamps without filtering.
        if len(config["coadds"]) > 0:
            stack.copy_to_gpu()
            append_coadds(keep, stack, config["coadds"], config["stamp_radius"])
            stack.clear_from_gpu()

        # Extract all the stamps for all time steps and append them onto the result rows.
        if config["save_all_stamps"]:
            append_all_stamps(keep, stack, config["stamp_radius"])

        logger.info(f"Found {len(keep)} potential trajectories.")

        # Save the results in as an ecsv file and/or a legacy text file.
        if config["legacy_filename"] is not None:
            logger.info(f"Saving legacy results to {config['legacy_filename']}")
            keep.write_trajectory_file(config["legacy_filename"])

        if config["result_filename"] is not None:
            logger.info(f"Saving results table to {config['result_filename']}")
            if not config["save_all_stamps"]:
                keep.write_table(config["result_filename"], cols_to_drop=["all_stamps"])
            else:
                keep.write_table(config["result_filename"])
        full_timer.stop()

        return keep

    def run_search_from_work_unit(self, work):
        """Run a KBMOD search from a WorkUnit object.

        Parameters
        ----------
        work : `WorkUnit`
            The input data and configuration.

        Returns
        -------
        keep : `Results`
            The results.
        """
        trj_generator = create_trajectory_generator(work.config, work_unit=work)

        # Run the search.
        return self.run_search(work.config, work.im_stack, trj_generator=trj_generator)
