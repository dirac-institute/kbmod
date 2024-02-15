import os
import time
import warnings

import astropy.coordinates as astroCoords
import astropy.units as u
import koffi
import numpy as np
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from numpy.linalg import lstsq

import kbmod.search as kb

from .analysis_utils import PostProcess
from .configuration import SearchConfiguration
from .data_interface import load_input_from_config, load_input_from_file
from .filters.sigma_g_filter import SigmaGClipping
from .filters.stamp_filters import append_all_stamps, get_coadds_and_filter
from .masking import apply_mask_operations
from .result_list import *
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

    def do_gpu_search(self, config, search):
        """
        Performs search on the GPU.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        search : `StackSearch`
            The C++ object that holds data and does searching.

        Returns
        -------
        search : `StackSearch`
            The C++ object holding the data and results.
        """
        width = search.get_image_width()
        height = search.get_image_height()
        ang_lim = self.get_angle_limits(config)
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
        if debug:
            print(f"Average Angle = {config['average_angle']}")
            print(f"Search Angle Limits = {ang_lim}")
            print(f"Velocity Limits = {config['v_arr']}")

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

        search.search(
            int(config["ang_arr"][2]),
            int(config["v_arr"][2]),
            ang_lim[0],
            ang_lim[1],
            config["v_arr"][0],
            config["v_arr"][1],
            int(config["num_obs"]),
        )

        search_timer.stop()
        return search

    def run_search(self, config, stack):
        """This function serves as the highest-level python interface for starting
        a KBMOD search given an ImageStack and SearchConfiguration.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        stack : `ImageStack`
            The stack before the masks have been applied. Modified in-place.

        Returns
        -------
        keep : ResultList
            The results.
        """
        full_timer = kb.DebugTimer("KBMOD", config["debug"])

        # Collect the MJDs.
        mjds = []
        for i in range(stack.img_count()):
            mjds.append(stack.get_obstime(i))

        # Set up the post processing data structure.
        kb_post_process = PostProcess(config, mjds)

        # Apply the mask to the images.
        if config["do_mask"]:
            stack = apply_mask_operations(config, stack)

        # Perform the actual search.
        search = kb.StackSearch(stack)
        search = self.do_gpu_search(config, search)

        # Load the KBMOD results into Python and apply a filter based on 'filter_type'.
        keep = kb_post_process.load_and_filter_results(
            search,
            config["lh_level"],
            chunk_size=config["chunk_size"],
            max_lh=config["max_lh"],
        )
        if config["do_stamp_filter"]:
            stamp_timer = kb.DebugTimer("stamp filtering", config["debug"])
            get_coadds_and_filter(
                keep,
                search.get_imagestack(),
                config,
                debug=config["debug"],
            )
            stamp_timer.stop()

        if config["do_clustering"]:
            cluster_timer = kb.DebugTimer("clustering", config["debug"])
            cluster_params = {}
            cluster_params["x_size"] = stack.get_width()
            cluster_params["y_size"] = stack.get_height()
            cluster_params["vel_lims"] = config["v_arr"]
            cluster_params["ang_lims"] = self.get_angle_limits(config)
            cluster_params["mjd"] = np.array(mjds)
            kb_post_process.apply_clustering(keep, cluster_params)
            cluster_timer.stop()

        # Extract all the stamps for all time steps and append them onto the result rows.
        if config["save_all_stamps"]:
            stamp_timer = kb.DebugTimer("computing all stamps", config["debug"])
            append_all_stamps(keep, search.get_imagestack(), config["stamp_radius"])
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
                work.config.set("average_angle", self._calc_suggested_angle(work.get_wcs(0), center_pixel))
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

    def _calc_suggested_angle(self, wcs, center_pixel=(1000, 2000), step=12):
        """Projects an unit-vector parallel with the ecliptic onto the image
        and calculates the angle of the projected unit-vector in the pixel
        space.

        Parameters
        ----------
        wcs : ``astropy.wcs.WCS``
            World Coordinate System object.
        center_pixel : tuple, array-like
            Pixel coordinates of image center.
        step : ``float`` or ``int``
            Size of step, in arcseconds, used to find the pixel coordinates of
                the second pixel in the image parallel to the ecliptic.

        Returns
        -------
        suggested_angle : ``float``
            Angle the projected unit-vector parallel to the ecliptic
            closes with the image axes. Used to transform the specified
            search angles, with respect to the ecliptic, to search angles
            within the image.

        Note
        ----
        It is not neccessary to calculate this angle for each image in an
        image set if they have all been warped to a common WCS.
        """
        # pick a starting pixel approximately near the center of the image
        # convert it to ecliptic coordinates
        start_pixel = np.array(center_pixel)
        start_pixel_coord = astroCoords.SkyCoord.from_pixel(start_pixel[0], start_pixel[1], wcs)
        start_ecliptic_coord = start_pixel_coord.geocentrictrueecliptic

        # pick a guess pixel by moving parallel to the ecliptic
        # convert it to pixel coordinates for the given WCS
        guess_ecliptic_coord = astroCoords.SkyCoord(
            start_ecliptic_coord.lon + step * u.arcsec,
            start_ecliptic_coord.lat,
            frame="geocentrictrueecliptic",
        )
        guess_pixel_coord = guess_ecliptic_coord.to_pixel(wcs)

        # calculate the distance, in pixel coordinates, between the guess and
        # the start pixel. Calculate the angle that represents in the image.
        x_dist, y_dist = np.array(guess_pixel_coord) - start_pixel
        return np.arctan2(y_dist, x_dist)
