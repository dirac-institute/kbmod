import time
import warnings

import numpy as np
from .analysis_utils import Interface, PostProcess
from .image_info import *
import kbmod.search as kb
from .known_objects import *
from .result_list import *


class run_search:
    """
    Run the KBMoD grid search.

    Parameters
    ----------
    input_parameters : `dict`
        Input parameters. Merged with the `defaults` dictionary.
        Must contain `im_filepath` and `res_filepath` keys, paths to
        the image and results directory respectively. Should contain
        `v_arr`, and `ang_arr`, which are lists containing the lower
        and upper velocity and angle limits.

    Attributes
    ----------
    default_mask_bits_dict : `dict`
        Map between mask key and bit values.
    default_flag_keys : `list`
        Pixels marked with these flags will be ignored in the search.
        Default: `["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]`
    default_repeated_flag_keys : `list`
        Don't know
    config : `dict`
        Search parameters.
    """
    
    def __init__(self, input_parameters):
        default_mask_bits_dict = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }
        default_flag_keys = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
        default_repeated_flag_keys = []
        defaults = {  # Mandatory values
            "im_filepath": None,
            "res_filepath": None,
            "time_file": None,
            "psf_file": None,
            # Suggested values
            "v_arr": [92.0, 526.0, 256],
            "ang_arr": [np.pi / 15, np.pi / 15, 128],
            # Optional values
            "output_suffix": "search",
            "mjd_lims": None,
            "average_angle": None,
            "do_mask": True,
            "mask_num_images": 2,
            "mask_threshold": None,
            "mask_grow": 10,
            "lh_level": 10.0,
            "psf_val": 1.4,
            "num_obs": 10,
            "num_cores": 1,
            "visit_in_filename": [0, 6],
            "sigmaG_lims": [25, 75],
            "chunk_size": 500000,
            "max_lh": 1000.0,
            "filter_type": "clipped_sigmaG",
            "center_thresh": 0.00,
            "peak_offset": [2.0, 2.0],
            "mom_lims": [35.5, 35.5, 2.0, 0.3, 0.3],
            "stamp_type": "sum",
            "stamp_radius": 10,
            "eps": 0.03,
            "gpu_filter": False,
            "do_clustering": True,
            "do_stamp_filter": True,
            "clip_negative": False,
            "sigmaG_filter_type": "lh",
            "cluster_type": "all",
            "cluster_function": "DBSCAN",
            "mask_bits_dict": default_mask_bits_dict,
            "flag_keys": default_flag_keys,
            "repeated_flag_keys": default_repeated_flag_keys,
            "bary_dist": None,
            "encode_psi_bytes": -1,
            "encode_phi_bytes": -1,
            "known_obj_thresh": None,
            "known_obj_jpl": False,
        }
        # Make sure input_parameters contains valid input options
        for key, val in input_parameters.items():
            if key in defaults:
                defaults[key] = val
            else:
                warnings.warn('Key "{}" is not a valid option. It is being ignored.'.format(key))
        self.config = defaults
        if self.config["im_filepath"] is None:
            raise ValueError("Image filepath not set")
        if self.config["res_filepath"] is None:
            raise ValueError("Results filepath not set")
        return

    def do_gpu_search(self, search, img_info, ec_angle, post_process):
        """
        Performs search on the GPU.

        Parameters
        ----------
        search : `kbmod.search.Search`
            Search object.
        img_info : `kbmod.search.ImageInfo`
            ImageInfo object.
        ec_angle : `float`
            Angle a 12 arcsecond segment parallel to the ecliptic is
            seen under from the image origin.
        post_process :
            Don't know
        """
        search_params = {}

        # Run the grid search
        # Set min and max values for angle and velocity
        if self.config["average_angle"] == None:
            average_angle = ec_angle
        else:
            average_angle = self.config["average_angle"]
        ang_min = average_angle - self.config["ang_arr"][0]
        ang_max = average_angle + self.config["ang_arr"][1]
        vel_min = self.config["v_arr"][0]
        vel_max = self.config["v_arr"][1]
        search_params["ang_lims"] = [ang_min, ang_max]
        search_params["vel_lims"] = [vel_min, vel_max]

        # If we are using barycentric corrections, compute the parameters and
        # enable it in the search function.
        if "bary_dist" in self.config.keys() and self.config["bary_dist"] is not None:
            bary_corr = self._calc_barycentric_corr(img_info, self.config["bary_dist"])
            # print average barycentric velocity for debugging

            mjd_range = img_info.get_duration()
            bary_vx = bary_corr[-1, 0] / mjd_range
            bary_vy = bary_corr[-1, 3] / mjd_range
            bary_v = np.sqrt(bary_vx * bary_vx + bary_vy * bary_vy)
            bary_ang = np.arctan2(bary_vy, bary_vx)
            print("Average Velocity from Barycentric Correction", bary_v, "pix/day", bary_ang, "angle")
            search.enable_corr(bary_corr.flatten())

        search_start = time.time()
        print("Starting Search")
        print("---------------------------------------")
        param_headers = (
            "Ecliptic Angle",
            "Min. Search Angle",
            "Max Search Angle",
            "Min Velocity",
            "Max Velocity",
        )
        param_values = (ec_angle, *search_params["ang_lims"], *search_params["vel_lims"])
        for header, val in zip(param_headers, param_values):
            print("%s = %.4f" % (header, val))

        # If we are using gpu_filtering, enable it and set the parameters.
        if self.config["gpu_filter"]:
            print("Using in-line GPU sigmaG filtering methods", flush=True)
            self.config["sigmaG_coeff"] = post_process._find_sigmaG_coeff(self.config["sigmaG_lims"])
            search.enable_gpu_sigmag_filter(
                np.array(self.config["sigmaG_lims"]) / 100.0,
                self.config["sigmaG_coeff"],
                self.config["lh_level"],
            )

        # If we are using an encoded image representation on GPU, enable it and
        # set the parameters.
        if self.config["encode_psi_bytes"] > 0 or self.config["encode_phi_bytes"] > 0:
            search.enable_gpu_encoding(self.config["encode_psi_bytes"], self.config["encode_phi_bytes"])

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
        """
        This function serves as the highest-level python interface for
        starting a KBMOD search.
        """
        start = time.time()
        kb_interface = Interface()
        kb_post_process = PostProcess(self.config)

        # Load the PSF.
        default_psf = kb.psf(self.config["psf_val"])

        # Load images to search
        stack, img_info, ec_angle = kb_interface.load_images(
            self.config["im_filepath"],
            self.config["time_file"],
            self.config["psf_file"],
            self.config["mjd_lims"],
            self.config["visit_in_filename"],
            default_psf,
        )

        # Apply the mask to the images.
        if self.config["do_mask"]:
            stack = kb_post_process.apply_mask(
                stack,
                mask_num_images=self.config["mask_num_images"],
                mask_threshold=self.config["mask_threshold"],
                mask_grow=self.config["mask_grow"],
            )

        # Perform the actual search.
        search = kb.stack_search(stack)
        search, search_params = self.do_gpu_search(search, img_info, ec_angle, kb_post_process)

        # Create filtering parameters.
        filter_params = {}
        filter_params["sigmaG_filter_type"] = self.config["sigmaG_filter_type"]

        # Load the KBMOD results into Python and apply a filter based on
        # 'filter_type.
        mjds = np.array(img_info.get_all_mjd())
        keep = kb_post_process.load_results(
            search,
            mjds,
            filter_params,
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            filter_type=self.config["filter_type"],
            max_lh=self.config["max_lh"],
        )
        if self.config["do_stamp_filter"]:
            keep = kb_post_process.apply_stamp_filter(
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
            cluster_params["x_size"] = img_info.get_x_size()
            cluster_params["y_size"] = img_info.get_y_size()
            cluster_params["vel_lims"] = search_params["vel_lims"]
            cluster_params["ang_lims"] = search_params["ang_lims"]
            cluster_params["mjd"] = mjds

            keep = kb_post_process.apply_clustering(keep, cluster_params)
        keep = kb_post_process.get_all_stamps(keep, search, self.config["stamp_radius"])

        # Count how many known objects we found.
        if self.config["known_obj_thresh"]:
            self._count_known_matches(keep, img_info, search)

        del search
        # Save the results
        kb_interface.save_results(
            self.config["res_filepath"],
            self.config["output_suffix"],
            keep,
            mjds,
        )

        end = time.time()
        del keep
        print("Time taken for patch: ", end - start)

    def _count_known_matches(self, keep, img_info, search):
        """
        Look up the known objects that overlap the images and count how many
        are found among the results.

        Parameters
        ----------
        keep : `dict`
           The results dictionary as defined by :py:meth:`SharedTools.gen_results_dict()`
        img_info : `kbmod.search.InfoSet`
           InfoSet.
        search : `kbmod.search.stack_search`
           A stack_search object containing information about the search.
        """
        # Lookup the known objects using either SkyBoT or the JPL API.
        print("-----------------")
        known_objects = KnownObjects()
        if self.config["known_obj_jpl"]:
            print("Quering known objects from JPL")
            known_objects.jpl_query_known_objects_mult(img_info)
        else:
            print("Quering known objects from SkyBoT")
            known_objects.skybot_query_known_objects_mult(img_info)
        known_objects.filter_observations(self.config["num_obs"])

        num_found = known_objects.get_num_results()
        print(
            "Found %i objects with at least %i potential observations." % (num_found, self.config["num_obs"])
        )
        print("-----------------")

        # If we didn't find any known objects then return early.
        if num_found == 0:
            return

        # Extract a list of predicted positions for the final results.
        found_objects = []
        for index in keep["final_results"]:
            ppos = search.get_mult_traj_pos(keep["results"][index])
            sky_pos = img_info.pixels_to_skycoords(ppos)
            found_objects.append(sky_pos)

        # Count the matches between known and found objects.
        count = known_objects.count_known_objects_found(
            found_objects, self.config["known_obj_thresh"], self.config["num_obs"]
        )
        print("Found %i of %i known objects." % (count, num_found))

    # might make sense to move this to another class
    # TODO add option for specific observatory?
    def _calc_barycentric_corr(self, img_info, dist):
        """
        This function calculates the barycentric corrections between
        each image and the first.
        
        The barycentric correction is the shift in x,y pixel position expected for
        an object that is stationary in barycentric coordinates, at a barycentric
        radius of dist au. This function returns a linear fit to the barycentric
        correction as a function of position on the first image.

        Parameters
        ----------
        img_info : `kbmod.search.ImageInfo`
            ImageInfo
        dist : `float`
            Distance to object from barycenter in AU.
        """
        from astropy import units as u
        from astropy.coordinates import SkyCoord, get_body_barycentric, solar_system_ephemeris
        from astropy.time import Time
        from numpy.linalg import lstsq

        wcslist = [img_info.stats[i].wcs for i in range(img_info.num_images)]
        mjdlist = np.array(img_info.get_all_mjd())
        x_size = img_info.get_x_size()
        y_size = img_info.get_y_size()

        # make grid with observer-centric RA/DEC of first image
        xlist, ylist = np.mgrid[0:x_size, 0:y_size]
        xlist = xlist.flatten()
        ylist = ylist.flatten()
        cobs = wcslist[0].pixel_to_world(xlist, ylist)

        # convert this grid to barycentric x,y,z, assuming distance r
        # [obs_to_bary_wdist()]
        with solar_system_ephemeris.set("de432s"):
            obs_pos = get_body_barycentric("earth", Time(mjdlist[0], format="mjd"))
        cobs.representation_type = "cartesian"
        # barycentric distance of observer
        r2_obs = obs_pos.x * obs_pos.x + obs_pos.y * obs_pos.y + obs_pos.z * obs_pos.z
        # calculate distance r along line of sight that gives correct
        # barycentric distance
        # |obs_pos + r * cobs|^2 = dist^2
        # obs_pos^2 + 2r (obs_pos dot cobs) + cobs^2 = dist^2
        dot = obs_pos.x * cobs.x + obs_pos.y * cobs.y + obs_pos.z * cobs.z
        bary_dist = dist * u.au
        r = -dot + np.sqrt(bary_dist * bary_dist - r2_obs + dot * dot)
        # barycentric coordinate is observer position + r * line of sight
        cbary = SkyCoord(
            obs_pos.x + r * cobs.x,
            obs_pos.y + r * cobs.y,
            obs_pos.z + r * cobs.z,
            representation_type="cartesian",
        )

        baryCoeff = np.zeros((len(wcslist), 6))
        for i in range(1, len(wcslist)):  # corections for wcslist[0] are 0
            # hold the barycentric coordinates constant and convert to new frame
            # by subtracting the observer's new position and converting to RA/DEC and pixel
            # [bary_to_obs_fast()]
            with solar_system_ephemeris.set("de432s"):
                obs_pos = get_body_barycentric("earth", Time(mjdlist[i], format="mjd"))
            c = SkyCoord(
                cbary.x - obs_pos.x, cbary.y - obs_pos.y, cbary.z - obs_pos.z, representation_type="cartesian"
            )
            c.representation_type = "spherical"
            pix = wcslist[i].world_to_pixel(c)

            # do linear fit to get coefficients
            ones = np.ones_like(xlist)
            A = np.stack([ones, xlist, ylist], axis=-1)
            coef_x, _, _, _ = lstsq(A, (pix[0] - xlist))
            coef_y, _, _, _ = lstsq(A, (pix[1] - ylist))
            baryCoeff[i, 0:3] = coef_x
            baryCoeff[i, 3:6] = coef_y

        return baryCoeff
