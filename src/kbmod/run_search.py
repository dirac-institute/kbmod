import os
import time
import warnings

import koffi
import numpy as np

import kbmod.search as kb

from .configuration import SearchConfiguration
from .data_interface import load_input_from_config, load_input_from_file
from .filters.clustering_filters import apply_clustering
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .filters.stamp_filters import append_all_stamps, get_coadds_and_filter
from .filters.stats_filters import CombinedStatsFilter
from .masking import apply_mask_operations
from .result_list import *
from .trajectory_generator import KBMODV1Search
from .wcs_utils import calc_ecliptic_angle
from .work_unit import WorkUnit


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
        keep : `ResultList`
            A ResultList object containing values from trajectories.
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
        num_cores = config["num_cores"]
        if num_cores <= 0:
            raise ValueError(f"Invalid number of cores {num_cores}")

        # Set up the list of results.
        img_stack = search.get_imagestack()
        num_times = img_stack.img_count()
        mjds = [img_stack.get_obstime(t) for t in range(num_times)]
        keep = ResultList(mjds)

        # Set up the clipped sigmaG filter.
        if sigmaG_lims is not None:
            bnds = sigmaG_lims
        else:
            bnds = [25, 75]
        clipper = SigmaGClipping(bnds[0], bnds[1], 2, clip_negative)

        # Set up the combined stats filter.
        if lh_level > 0.0:
            stats_filter = CombinedStatsFilter(min_obs=num_obs, min_lh=lh_level)
        else:
            stats_filter = CombinedStatsFilter(min_obs=num_obs)

        print("---------------------------------------")
        print("Retrieving Results")
        print("---------------------------------------")
        likelihood_limit = False
        res_num = 0
        total_count = 0
        while likelihood_limit is False:
            print("Getting results...")
            results = search.get_results(res_num, chunk_size)
            print("---------------------------------------")
            print("Chunk Start = %i" % res_num)
            print("Chunk Max Likelihood = %.2f" % results[0].lh)
            print("Chunk Min. Likelihood = %.2f" % results[-1].lh)
            print("---------------------------------------")

            result_batch = ResultList(mjds)
            for i, trj in enumerate(results):
                # Stop as soon as we hit a result below our limit, because anything after
                # that is not guarrenteed to be valid due to potential on-GPU filtering.
                if trj.lh < lh_level:
                    likelihood_limit = True
                    break

                if trj.lh < max_lh:
                    row = ResultRow(trj, num_times)
                    psi_curve = np.array(search.get_psi_curves(trj))
                    phi_curve = np.array(search.get_phi_curves(trj))
                    row.set_psi_phi(psi_curve, phi_curve)
                    result_batch.append_result(row)
                    total_count += 1

            batch_size = result_batch.num_results()
            print("Extracted batch of %i results for total of %i" % (batch_size, total_count))
            if batch_size > 0:
                apply_clipped_sigma_g(clipper, result_batch, num_cores)
                result_batch.apply_filter(stats_filter)

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
        keep : `ResultList`
            The results.
        """
        # Create the search object which will hold intermediate data and results.
        search = kb.StackSearch(stack)

        width = search.get_image_width()
        height = search.get_image_height()
        debug = config["debug"]

        # Set the search bounds.
        if config["x_pixel_bounds"] and len(config["x_pixel_bounds"]) == 2:
            search.set_start_bounds_x(config["x_pixel_bounds"][0], config["x_pixel_bounds"][1])
        elif config["x_pixel_buffer"] and config["x_pixel_buffer"] > 0:
            search.set_start_bounds_x(-config["x_pixel_buffer"], width + config["x_pixel_buffer"])

        if config["y_pixel_bounds"] and len(config["y_pixel_bounds"]) == 2:
            search.set_start_bounds_y(config["y_pixel_bounds"][0], config["y_pixel_bounds"][1])
        elif config["y_pixel_buffer"] and config["y_pixel_buffer"] > 0:
            search.set_start_bounds_y(-config["y_pixel_buffer"], height + config["y_pixel_buffer"])

        search_timer = kb.DebugTimer("Grid Search", debug)

        # If we are using gpu_filtering, enable it and set the parameters.
        if config["gpu_filter"]:
            if debug:
                print("Using in-line GPU sigmaG filtering methods", flush=True)
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

        # Enable debugging.
        if config["debug"]:
            search.set_debug(config["debug"])

        # Do the actual search.
        candidates = [trj for trj in trj_generator]
        search.search(candidates, int(config["num_obs"]))
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
        keep : ResultList
            The results.
        """
        full_timer = kb.DebugTimer("KBMOD", config["debug"])

        # Apply the mask to the images.
        if config["do_mask"]:
            stack = apply_mask_operations(config, stack)

        # Perform the actual search.
        if trj_generator is None:
            ang_limits = self.get_angle_limits(config)
            trj_generator = KBMODV1Search(
                int(config["v_arr"][2]),
                config["v_arr"][0],
                config["v_arr"][1],
                int(config["ang_arr"][2]),
                ang_limits[0],
                ang_limits[1],
            )
        keep = self.do_gpu_search(config, stack, trj_generator)

        if config["do_stamp_filter"]:
            stamp_timer = kb.DebugTimer("stamp filtering", config["debug"])
            get_coadds_and_filter(
                keep,
                stack,
                config,
                debug=config["debug"],
            )
            stamp_timer.stop()

        if config["do_clustering"]:
            cluster_timer = kb.DebugTimer("clustering", config["debug"])
            mjds = [stack.get_obstime(t) for t in range(stack.img_count())]
            cluster_params = {
                "ang_lims": self.get_angle_limits(config),
                "cluster_type": config["cluster_type"],
                "eps": config["eps"],
                "mjd": np.array(mjds),
                "vel_lims": config["v_arr"],
                "width": stack.get_width(),
                "height": stack.get_height(),
            }
            apply_clustering(keep, cluster_params)
            cluster_timer.stop()

        # Extract all the stamps for all time steps and append them onto the result rows.
        if config["save_all_stamps"]:
            stamp_timer = kb.DebugTimer("computing all stamps", config["debug"])
            append_all_stamps(keep, stack, config["stamp_radius"])
            stamp_timer.stop()

        # TODO - Re-enable the known object counting once we have a way to pass
        # A WCS into the WorkUnit.
        # Count how many known objects we found.
        # if config["known_obj_thresh"]:
        #    _count_known_matches(keep, search)

        # Save the results and the configuration information used.
        print(f"Found {keep.num_results()} potential trajectories.")
        if config["res_filepath"] is not None and config["ind_output_files"]:
            keep.save_to_files(config["res_filepath"], config["output_suffix"])

            config_filename = os.path.join(config["res_filepath"], f"config_{config['output_suffix']}.yml")
            config.to_file(config_filename, overwrite=True)
        if config["result_filename"] is not None:
            keep.write_table(config["result_filename"], keep_all_stamps=config["save_all_stamps"])

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
        keep : ResultList
            The results.
        """
        # Set the average angle if it is not set.
        if work.config["average_angle"] is None:
            center_pixel = (work.im_stack.get_width() / 2, work.im_stack.get_height() / 2)
            if work.get_wcs(0) is not None:
                work.config.set("average_angle", calc_ecliptic_angle(work.get_wcs(0), center_pixel))
            else:
                print("WARNING: average_angle is unset and no WCS provided. Using 0.0.")
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
        keep : ResultList
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
        keep : ResultList
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

        print("-----------------")
        matches = {}
        known_obj_thresh = config["known_obj_thresh"]
        min_obs = config["known_obj_obs"]
        if config["known_obj_jpl"]:
            print("Quering known objects from JPL")
            matches = koffi.jpl_query_known_objects_stack(
                potential_sources=ps_list,
                images=metadata,
                min_observations=min_obs,
                tolerance=known_obj_thresh,
            )
        else:
            print("Quering known objects from SkyBoT")
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
        print("Found %i objects with at least %i potential observations." % (num_found, config["num_obs"]))

        if num_found > 0:
            print(matches_string)
        print("-----------------")
