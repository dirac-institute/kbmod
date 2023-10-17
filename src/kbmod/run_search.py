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

from .analysis_utils import find_sigmaG_coeff, PostProcess
from .data_interface import load_input_from_config
from .configuration import SearchConfiguration
from .masking import (
    BitVectorMasker,
    DictionaryMasker,
    GlobalDictionaryMasker,
    GrowMask,
    ThresholdMask,
    apply_mask_operations,
)
from .result_list import *


class run_search:
    """
    Run the KBMOD grid search.

    Parameters
    ----------
    input_parameters : ``dict``
        Additional parameters. Merged with (and checked against) the loaded input file and
        the defaults provided in the SearchConfiguration class.
    config_file : ``str`` (optional)
        The name and path of the configuration file.

    Attributes
    ----------
    config : ``SearchConfiguration``
        Search parameters.
    """

    def __init__(self, input_parameters, config_file=None):
        # Load parameters from a file.
        if config_file != None:
            self.config = SearchConfiguration.from_file(config_file)
        else:
            self.config = SearchConfiguration()

        # Load any additional parameters (overwriting what is there).
        if len(input_parameters) > 0:
            for key, value in input_parameters.items():
                self.config.set(key, value)

        # Validate the configuration.
        self.config.validate()

    def do_masking(self, stack):
        """Perform the masking based on the search's configuration parameters.

        Parameters
        ----------
        stack : `kbmod.ImageStack`
            The stack before the masks have been applied.
        """
        mask_steps = []

        # Prioritize the mask_bit_vector over the dictionary based version.
        if self.config["mask_bit_vector"]:
            mask_steps.append(BitVectorMasker(self.config["mask_bit_vector"], [0]))
        elif self.config["flag_keys"] and len(self.config["flag_keys"]) > 0:
            mask_steps.append(DictionaryMasker(self.config["mask_bits_dict"], self.config["flag_keys"]))

        # Add the threshold mask if it is set.
        if self.config["mask_threshold"]:
            mask_steps.append(ThresholdMask(self.config["mask_threshold"]))

        # Add the global masking if it is set.
        if self.config["repeated_flag_keys"] and len(self.config["repeated_flag_keys"]) > 0:
            mask_steps.append(
                GlobalDictionaryMasker(
                    self.config["mask_bits_dict"],
                    self.config["repeated_flag_keys"],
                    self.config["mask_num_images"],
                )
            )

        # Grow the mask.
        if self.config["mask_grow"] and self.config["mask_grow"] > 0:
            mask_steps.append(GrowMask(self.config["mask_grow"]))

        # Apply the masks.
        stack = apply_mask_operations(stack, mask_steps)

        return stack

    def do_gpu_search(self, search):
        """
        Performs search on the GPU.

        Parameters
        ----------
        search : ``~kbmod.search.Search``
            Search object.
        """
        width = search.get_image_width()
        height = search.get_image_height()
        search_params = {}

        # Run the grid search
        # Set min and max values for angle and velocity
        ang_min = self.config["average_angle"] - self.config["ang_arr"][0]
        ang_max = self.config["average_angle"] + self.config["ang_arr"][1]
        vel_min = self.config["v_arr"][0]
        vel_max = self.config["v_arr"][1]
        search_params["ang_lims"] = [ang_min, ang_max]
        search_params["vel_lims"] = [vel_min, vel_max]

        # Set the search bounds.
        if self.config["x_pixel_bounds"] and len(self.config["x_pixel_bounds"]) == 2:
            search.set_start_bounds_x(self.config["x_pixel_bounds"][0], self.config["x_pixel_bounds"][1])
        elif self.config["x_pixel_buffer"] and self.config["x_pixel_buffer"] > 0:
            search.set_start_bounds_x(-self.config["x_pixel_buffer"], width + self.config["x_pixel_buffer"])

        if self.config["y_pixel_bounds"] and len(self.config["y_pixel_bounds"]) == 2:
            search.set_start_bounds_y(self.config["y_pixel_bounds"][0], self.config["y_pixel_bounds"][1])
        elif self.config["y_pixel_buffer"] and self.config["y_pixel_buffer"] > 0:
            search.set_start_bounds_y(-self.config["y_pixel_buffer"], height + self.config["y_pixel_buffer"])

        search_start = time.time()
        print("Starting Search")
        print("---------------------------------------")
        print(f"Average Angle = {self.config['average_angle']}")
        print(f"Search Angle Limits = {search_params['ang_lims']}")
        print(f"Velocity Limits = {search_params['vel_lims']}")

        # If we are using gpu_filtering, enable it and set the parameters.
        if self.config["gpu_filter"]:
            print("Using in-line GPU sigmaG filtering methods", flush=True)
            coeff = find_sigmaG_coeff(self.config["sigmaG_lims"])
            search.enable_gpu_sigmag_filter(
                np.array(self.config["sigmaG_lims"]) / 100.0,
                coeff,
                self.config["lh_level"],
            )

        # If we are using an encoded image representation on GPU, enable it and
        # set the parameters.
        if self.config["encode_psi_bytes"] > 0 or self.config["encode_phi_bytes"] > 0:
            search.enable_gpu_encoding(self.config["encode_psi_bytes"], self.config["encode_phi_bytes"])

        # Enable debugging.
        if self.config["debug"]:
            search.set_debug(self.config["debug"])

        search.search(
            int(self.config["ang_arr"][2]),
            int(self.config["v_arr"][2]),
            *search_params["ang_lims"],
            *search_params["vel_lims"],
            int(self.config["num_obs"]),
        )
        print("Search finished in {0:.3f}s".format(time.time() - search_start), flush=True)
        return (search, search_params)

    def run_search(self):
        """This function serves as the highest-level python interface for starting
        a KBMOD search.

        The `config` attribute requires the following key value pairs.

        Parameters
        ----------
        self.config.im_filepath : string
            Path to the folder containing the images to be ingested into
            KBMOD and searched over.
        self.config.res_filepath : string
            Path to the folder that will contain the results from the search.
            If ``None`` the program skips outputting the files.
        self.config.out_suffix : string
            Suffix to append to the output files. Used to differentiate
            between different searches over the same stack of images.
        self.config.time_file : string
            Path to the file containing the image times (or None to use
            values from the FITS files).
        self.config.psf_file : string
            Path to the file containing the image PSFs (or None to use default).
        self.config.lh_level : float
            Minimum acceptable likelihood level for a trajectory.
            Trajectories with likelihoods below this value will be discarded.
        self.config.psf_val : float
            The value of the variance of the default PSF to use.
        self.config.mjd_lims : numpy array
            Limits the search to images taken within the limits input by
            mjd_lims (or None for no filtering).
        self.config.average_angle : float
            Overrides the ecliptic angle calculation and instead centers
            the average search around average_angle.

        Returns
        -------
        keep : ResultList
            The results.
        """
        start = time.time()

        # Load images to search
        stack, wcs_list, mjds = load_input_from_config(self.config, verbose=self.config["debug"])

        # Compute the suggested search angle from the images. This is a 12 arcsecond
        # segment parallel to the ecliptic is seen under from the image origin.
        if self.config["average_angle"] == None:
            center_pixel = (stack.get_width() / 2, stack.get_height() / 2)
            self.config.set("average_angle", self._calc_suggested_angle(wcs_list[0], center_pixel))

        # Set up the post processing data structure.
        kb_post_process = PostProcess(self.config, mjds)

        # Apply the mask to the images.
        if self.config["do_mask"]:
            stack = self.do_masking(stack)

        # Perform the actual search.
        search = kb.StackSearch(stack)
        search, search_params = self.do_gpu_search(search)

        # Load the KBMOD results into Python and apply a filter based on
        # 'filter_type.
        mjds = np.array(mjds)
        keep = kb_post_process.load_and_filter_results(
            search,
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            max_lh=self.config["max_lh"],
        )
        if self.config["do_stamp_filter"]:
            kb_post_process.apply_stamp_filter(
                keep,
                search,
                center_thresh=self.config["center_thresh"],
                peak_offset=self.config["peak_offset"],
                mom_lims=self.config["mom_lims"],
                stamp_type=self.config["stamp_type"],
                stamp_radius=self.config["stamp_radius"],
            )

        if self.config["do_clustering"]:
            cluster_params = {}
            cluster_params["x_size"] = stack.get_width()
            cluster_params["y_size"] = stack.get_height()
            cluster_params["vel_lims"] = search_params["vel_lims"]
            cluster_params["ang_lims"] = search_params["ang_lims"]
            cluster_params["mjd"] = mjds
            kb_post_process.apply_clustering(keep, cluster_params)

        # Extract all the stamps.
        kb_post_process.get_all_stamps(keep, search, self.config["stamp_radius"])

        # Count how many known objects we found.
        if self.config["known_obj_thresh"]:
            self._count_known_matches(keep, search)

        del search

        # Save the results and the configuration information used.
        print(f"Found {keep.num_results()} potential trajectories.")
        if self.config["res_filepath"] is not None:
            keep.save_to_files(self.config["res_filepath"], self.config["output_suffix"])

            config_filename = os.path.join(
                self.config["res_filepath"], f"config_{self.config['output_suffix']}.yml"
            )
            self.config.to_file(config_filename, overwrite=True)

        end = time.time()
        print("Time taken for patch: ", end - start)

        return keep

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
        im_filepath = self.config["im_filepath"]
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
        known_obj_thresh = self.config["known_obj_thresh"]
        min_obs = self.config["known_obj_obs"]
        if self.config["known_obj_jpl"]:
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
        print(
            "Found %i objects with at least %i potential observations." % (num_found, self.config["num_obs"])
        )

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

        See Also
        --------
        run_search.do_gpu_search
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
