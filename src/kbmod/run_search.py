import os
import time

import koffi
import numpy as np

import kbmod.search as kb

from .configuration import SearchConfiguration
from .data_interface import load_input_from_config, load_input_from_file
from .filters.clustering_filters import apply_clustering
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .filters.stamp_filters import append_all_stamps, append_coadds, get_coadds_and_filter_results
from .masking import apply_mask_operations
from .results import Results
from .trajectory_generator import create_trajectory_generator, KBMODV1SearchConfig
from .wcs_utils import calc_ecliptic_angle
from .work_unit import WorkUnit


logger = kb.Logging.getLogger(__name__)


class SearchRunner:
    """A class to run the KBMOD grid search."""

    def __init__(self):
        pass

    def get_angle_limits(self, config):
        """Compute the angle limits based on the configuration information.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters

        Returns
        -------
        res : `list`
            A list with the minimum and maximum angle to search (in pixel space).
        """
        ang_min = config["average_angle"] - config["ang_arr"][0]
        ang_max = config["average_angle"] + config["ang_arr"][1]
        return [ang_min, ang_max]

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

        logger.info("Retrieving Results")
        likelihood_limit = False
        res_num = 0
        total_count = 0
        while likelihood_limit is False:
            logger.info(f"Chunk Start = {res_num} (size={chunk_size})")
            results = search.get_results(res_num, chunk_size)
            logger.info(f"Chunk Max Likelihood = {results[0].lh}")
            logger.info(f"Chunk Min. Likelihood = {results[-1].lh}")

            trj_batch = []
            psi_batch = []
            phi_batch = []
            for i, trj in enumerate(results):
                # Stop as soon as we hit a result below our limit, because anything after
                # that is not guarrenteed to be valid due to potential on-GPU filtering.
                if trj.lh < lh_level:
                    likelihood_limit = True
                    break

                if trj.lh < max_lh:
                    trj_batch.append(trj)
                    psi_batch.append(search.get_psi_curves(trj))
                    phi_batch.append(search.get_phi_curves(trj))
                    total_count += 1

            batch_size = len(trj_batch)
            logger.info(f"Extracted batch of {batch_size} results for total of {total_count}")

            if batch_size > 0:
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
            If None uses the default KBMODv1 grid search

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
            stack = apply_mask_operations(config, stack)

        # Perform the actual search.
        if trj_generator is None:
            trj_generator = create_trajectory_generator(config)
        keep = self.do_gpu_search(config, stack, trj_generator)

        if config["do_stamp_filter"]:
            stack.copy_to_gpu()
            get_coadds_and_filter_results(keep, stack, config)
            stack.clear_from_gpu()

        if config["do_clustering"]:
            cluster_timer = kb.DebugTimer("clustering", logger)
            mjds = [stack.get_obstime(t) for t in range(stack.img_count())]
            cluster_params = {
                "ang_lims": self.get_angle_limits(config),
                "cluster_type": config["cluster_type"],
                "eps": config["eps"],
                "times": np.array(mjds),
                "vel_lims": config["v_arr"],
                "width": stack.get_width(),
                "height": stack.get_height(),
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

        # TODO - Re-enable the known object counting once we have a way to pass
        # A WCS into the WorkUnit.
        # Count how many known objects we found.
        # if config["known_obj_thresh"]:
        #    _count_known_matches(keep, search)

        # Save the results and the configuration information used.
        logger.info(f"Found {len(keep)} potential trajectories.")
        if config["res_filepath"] is not None and config["ind_output_files"]:
            trj_filename = os.path.join(config["res_filepath"], f"results_{config['output_suffix']}.txt")
            keep.write_trajectory_file(trj_filename)

            config_filename = os.path.join(config["res_filepath"], f"config_{config['output_suffix']}.yml")
            config.to_file(config_filename, overwrite=True)

            stats_filename = os.path.join(
                config["res_filepath"], f"filter_stats_{config['output_suffix']}.csv"
            )
            keep.write_filtered_stats(stats_filename)

            if "all_stamps" in keep.colnames:
                keep.write_column("all_stamps", f"all_stamps_{config['output_suffix']}.npy")

        if config["result_filename"] is not None:
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
        # Set the average angle if it is not set.
        if work.config["average_angle"] is None:
            center_pixel = (work.im_stack.get_width() / 2, work.im_stack.get_height() / 2)
            if work.get_wcs(0) is not None:
                work.config.set("average_angle", calc_ecliptic_angle(work.get_wcs(0), center_pixel))
            else:
                logger.warning("Average angle not set and no WCS provided. Setting average_angle=0.0")
                work.config.set("average_angle", 0.0)

        # Run the search.
        return self.run_search(work.config, work.im_stack)

    def run_search_from_config(self, config):
        """Run a KBMOD search from a SearchConfiguration object
        (or corresponding dictionary).

        Parameters
        ----------
        config : `SearchConfiguration` or `dict`
            The configuration object with all the information for the run.

        Returns
        -------
        keep : `Results`
            The results.
        """
        if type(config) is dict:
            config = SearchConfiguration.from_dict(config)

        # Load the data.
        work = load_input_from_config(config)
        return self.run_search_from_work_unit(work)

    def run_search_from_file(self, filename, overrides=None):
        """Run a KBMOD search from a configuration or WorkUnit file.

        Parameters
        ----------
        filename : `str`
            The name of the input file.
        overrides : `dict`, optional
            A dictionary of configuration parameters to override. For testing.

        Returns
        -------
        keep : `Results`
            The results.
        """
        work = load_input_from_file(filename, overrides)
        return self.run_search_from_work_unit(work)

    def _count_known_matches(self, result_list, search):
        """Look up the known objects that overlap the images and count how many
        are found among the results.

        Parameters
        ----------
        result_list : ``kbmod.ResultList``
            The result objects found by the search.
        search : ``kbmod.search.StackSearch``
            A StackSearch object containing information about the search.
        """
        # Get the image metadata
        im_filepath = config["im_filepath"]
        filenames = sorted(os.listdir(im_filepath))
        image_list = [os.path.join(im_filepath, im_name) for im_name in filenames]
        metadata = koffi.ImageMetadataStack(image_list)

        # Get the pixel positions of results
        ps_list = []

        times = search.stack.build_zeroed_times()
        for row in result_list.results:
            trj = row.trajectory
            PixelPositions = [[trj.get_x_pos(t), trj.get_y_pos(t)] for t in times]

            ps = koffi.PotentialSource()
            ps.build_from_images_and_xy_positions(PixelPositions, metadata)
            ps_list.append(ps)

        matches = {}
        known_obj_thresh = config["known_obj_thresh"]
        min_obs = config["known_obj_obs"]
        if config["known_obj_jpl"]:
            logger.info("Querying known objects from JPL.")
            matches = koffi.jpl_query_known_objects_stack(
                potential_sources=ps_list,
                images=metadata,
                min_observations=min_obs,
                tolerance=known_obj_thresh,
            )
        else:
            logger.info("Querying known objects from SkyBoT.")
            matches = koffi.skybot_query_known_objects_stack(
                potential_sources=ps_list,
                images=metadata,
                min_observations=min_obs,
                tolerance=known_obj_thresh,
            )

        matches_string = ""
        num_found = 0
        for ps_id in matches.keys():
            if len(matches[ps_id]) > 0:
                num_found += 1
                matches_string += f"result id {ps_id}:" + str(matches[ps_id])[1:-1] + "\n"
        logger.info(f"Found {num_found} objects with at least {config['num_obs']} potential observations.")

        if num_found > 0:
            logger.info(f"{matches_string}")
