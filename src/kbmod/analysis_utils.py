import os
import csv
import time
import heapq
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import astropy.coordinates as astroCoords
from scipy.special import erfinv  # import mpmath
from sklearn.cluster import DBSCAN, OPTICS

from .image_info import *
from .result_list import *
import kbmod.search as kb


class Interface(SharedTools):
    """
    This class manages the KBMOD interface with the local filesystem, the cpp
    KBMOD code, and the PostProcess python filtering functions. It is
    responsible for loading in data from .fits files, initializing the kbmod
    object, loading results from the kbmod object into python, and saving
    results to file.
    """

    def __init__(self):
        return

    def load_images(
        self,
        im_filepath,
        time_file,
        psf_file,
        mjd_lims,
        visit_in_filename,
        default_psf,
    ):
        """
        This function loads images and ingests them into a search object.
        INPUT-
            im_filepath : string
                Image file path from which to load images
            time_file : string
                File name containing image times
            psf_file : string or None
                File name containing the image-specific PSFs. If set to None the code
                will use the provided default psf for all images.
            mjd_lims : int list
                Optional MJD limits on the images to search.
            visit_in_filename : int list
                A list containg the first and last character of the visit ID
                contained in the filename. By default, the first six characters
                of the filenames in this folder should contain the visit ID.
            default_psf : PointSpreadFunc object
                The default PSF in case no image-specific PSF is provided.
        OUTPUT-
            stack : kbmod.image_stack object
            img_info : image_info object
            ec_angle : float - the ecliptic angle for the images
        """
        img_info = ImageInfoSet()
        print("---------------------------------------")
        print("Loading Images")
        print("---------------------------------------")

        # Load a mapping from visit numbers to the visit times. This dictionary stays
        # empty if no time file is specified.
        image_time_dict = OrderedDict()
        if time_file:
            visit_nums, visit_times = np.genfromtxt(time_file, unpack=True)
            for visit_num, visit_time in zip(visit_nums, visit_times):
                image_time_dict[str(int(visit_num))] = visit_time

        # Load a mapping from visit numbers to PSFs. This dictionary stays
        # empty if no time file is specified.
        image_psf_dict = OrderedDict()
        if psf_file:
            visit_nums, psf_vals = np.genfromtxt(psf_file, unpack=True)
            for visit_num, visit_psf in zip(visit_nums, psf_vals):
                image_psf_dict[str(int(visit_num))] = visit_psf

        # Retrieve the list of visits (file names) in the data directory.
        patch_visits = sorted(os.listdir(im_filepath))

        # Get the bounds for the characters to use for the visit ID in the file name.
        id_start = visit_in_filename[0]
        id_end = visit_in_filename[1]

        # Load the images themselves.
        filenames = []
        images = []
        visit_times = []
        for visit_file in np.sort(patch_visits):
            full_file_path = "{0:s}/{1:s}".format(im_filepath, visit_file)
            visit_str = str(int(visit_file[id_start:id_end]))

            # Check if we can prune the file based on the timestamp. We do this
            # before the file load to save time, but might have to recheck if the
            # time stamp is stored in the file itself.
            time_stamp = -1.0
            if visit_str in image_time_dict:
                time_stamp = image_time_dict[visit_str]
                if mjd_lims is not None:
                    if time_stamp < mjd_lims[0] or time_stamp > mjd_lims[1]:
                        continue

            # Check if the image has a specific PSF.
            psf = default_psf
            if visit_str in image_psf_dict:
                psf = kb.psf(image_psf_dict[visit_str])

            # Load the image file.
            img = kb.layered_image(full_file_path, psf)

            # If we didn't previously load a time stamp, check whether the file contains
            # that information and retry the time based filteriing.
            if time_stamp <= 0.0:
                time_stamp = img.get_time()  # default of 0.0
                # Skip images without valid times.
                if time_stamp <= 0.0:
                    print("WARNING: No timestamp provided for visit %s. Skipping." % visit_str)
                    continue
                # Skip images with times outside the specified range.
                if mjd_lims is not None:
                    if time_stamp < mjd_lims[0] or time_stamp > mjd_lims[1]:
                        continue
            else:
                # If we have a valid timestamp from the file, use that for the image.
                img.set_time(time_stamp)

            # Save the file, time, and image information.
            filenames.append(full_file_path)
            images.append(img)
            visit_times.append(time_stamp)

        print("Loaded {0:d} images".format(len(images)))
        stack = kb.image_stack(images)

        # Load the additional image information, including
        # WCS and computing the ecliptic angles.
        img_info.load_image_info_from_files(filenames)

        # Create a list of visit times and visit times shifts to 0.0.
        img_info.set_times_mjd(np.array(visit_times))
        times = img_info.get_zero_shifted_times()
        stack.set_times(times)
        print("Times set", flush=True)

        # Compute the ecliptic angle for the images.
        center_pixel = (img_info.stats[0].width / 2, img_info.stats[0].height / 2)
        ec_angle = self._calc_ecliptic_angle(img_info.stats[0].wcs, center_pixel)

        return (stack, img_info, ec_angle)

    def save_results(self, res_filepath, out_suffix, keep, all_times):
        """
        This function saves results from a given search method (either region
        search or grid search)

        Parameters
        ----------
            res_filepath : string
            out_suffix : string
                Suffix to append to the output file name
            keep : ResultList
                ResultList containing the values to keep and print to file.
            all_times : list
                A list of times.
        """
        print("---------------------------------------")
        print("Saving Results")
        print("---------------------------------------", flush=True)
        keep.save_to_files(res_filepath, out_suffix)

    def _calc_ecliptic_angle(self, wcs, center_pixel=(1000, 2000), step=12):
        """
        Projects an unit-vector parallel with the ecliptic onto the image
        and calculates the angle of the projected unit-vector in the pixel
        space.

        Parameters
        ----------
        wcs : `astropy.wcs.WCS`
            World Coordinate System object.
        center_pixel : `tuple`, array-like
            Pixel coordinates of image center.
        step : `float` or `int`
            Size of step, in arcseconds, used to find the pixel
            coordinates of the second pixel in the image parallel to
            the ecliptic.

        Returns
        -------
        angle : `float`
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

    def _calc_barycentric_corr(self, wcslist, mjdlist, x_size, y_size, dist):
        """
        This function calculates the barycentric corrections between wcslist[0]
        and each frame in wcslist.
        The barycentric correction is the shift in x,y pixel position expected for
        an object that is stationary in barycentric coordinates, at a barycentric
        radius of dist au. This function returns a linear fit to the barycentric
        correction as a function of position on the image with wcs0.
        """

        # make grid with observer-centric RA/DEC of wcs0
        xlist, ylist = np.mgrid[0:x_size, 0:y_size]
        xlist = xlist.flatten()
        ylist = ylist.flatten()
        cobs = wcs0.pixel_to_world(xlist, ylist)

        # convert this grid to barycentric x,y,z, assuming distance r
        # [obs_to_bary_wdist()]
        with solar_system_ephemeris.set("de432s"):
            obs_pos = get_body_barycentric("earth", Time(mjdlist[0], format="mjd"))
        cobs.represention_type = "cartesian"
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
        cbary = SkyCoord(obs_pos.x + r * c.x, obs_pos.y + r * c.y, obs_pos.z + r * c.z)

        baryCoeff = np.zeros((len(wcslist), 6))
        for i in range(1, len(wcslist)):  # corections for wcslist[0] are 0
            # hold the barycentric coordinates constant and convert to new frame
            # by subtracting the observer's new position and converting to RA/DEC and pixel
            # [bary_to_obs_fast()]
            with solar_system.ephemeris.set("de432s"):
                obs_pos = get_body_barycentric("earth", Time(mjdlist[i], format="mjd"))
            c = SkyCoord(cbary.x - obs_pos.x, cbary.y - obs_pos.y, cbary.z - obs_pos.z)
            c.representation_type = "spherical"
            pix = wcslist[i].world_to_pixel(c)

            # do linear fit to get coefficients
            ones = np.ones_like(xlist)
            A = np.stack([ones, xlist, ylist], axis=-1)
            coef_x, _, _, _ = lstsq(A_x, (pix[0] - xlist))
            coef_y, _, _, _ = lstsq(A_y, (pix[1] - ylist))
            baryCoeff[i, 0:3] = coef_x
            baryCoeff[i, 3:6] = coef_y

        return baryCoeff


class PostProcess(SharedTools):
    """
    This class manages the post-processing utilities used to filter out and
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
        self.mask_bits_dict = config["mask_bits_dict"]
        self.flag_keys = config["flag_keys"]
        self.repeated_flag_keys = config["repeated_flag_keys"]
        self._mjds = mjds
        return

    def apply_mask(self, stack, mask_num_images=2, mask_threshold=None, mask_grow=10):
        """
        This function applys a mask to the images in a KBMOD stack. This mask
        sets a high variance for masked pixels
        INPUT-
            stack : kbmod.image_stack object
                The stack before the masks have been applied.
            mask_num_images : int
                The minimum number of images in which a masked pixel must
                appear in order for it to be masked out. E.g. if
                masked_num_images=2, then an object must appear in the same
                place in at least two images in order for the variance at that
                location to be increased.
            mask_threshold : float
                Any pixel with a flux greater than mask_threshold is masked out.
            mask_grow : integer
                The number of pixels by which to grow the mask.
        OUTPUT-
            stack : kbmod.image_stack object
                The stack after the masks have been applied.
        """
        mask_bits_dict = self.mask_bits_dict
        flag_keys = self.flag_keys
        global_flag_keys = self.repeated_flag_keys

        flags = 0
        for bit in flag_keys:
            flags += 2 ** mask_bits_dict[bit]

        flag_exceptions = [0]
        # mask any pixels which have any of these flags
        global_flags = 0
        for bit in global_flag_keys:
            global_flags += 2 ** mask_bits_dict[bit]

        # Apply masks if needed.
        if len(flag_keys) > 0:
            stack.apply_mask_flags(flags, flag_exceptions)
        if mask_threshold:
            stack.apply_mask_threshold(mask_threshold)
        if len(global_flag_keys) > 0:
            stack.apply_global_mask(global_flags, mask_num_images)

        # Grow the masks by 'mask_grow' pixels.
        stack.grow_mask(mask_grow, True)

        return stack

    def load_and_filter_results(
        self,
        search,
        filter_params,
        lh_level,
        filter_type="clipped_sigmaG",
        chunk_size=500000,
        max_lh=1e9,
    ):
        """
        This function loads results that are output by the gpu grid search.
        Results are loaded in chunks and evaluated to see if the minimum
        likelihood level has been reached. If not, another chunk of results is
        fetched.
        INPUT-
            search : kbmod search object
            filter_params : dictionary
                Contains optional filtering paramters.
            lh_level : float
                The minimum likelihood theshold for an acceptable result.
                Results below this likelihood level will be discarded.
            filter_type : string
                The type of initial filtering to apply. Acceptable values are
                'clipped_sigmaG'
                'clipped_average'
                'kalman'
            chunk_size : int
                The number of results to load at a given time from search.
            max_lh : float
                The maximum likelihood threshold for an acceptable results.
                Results ABOVE this likelihood level will be discarded.
        OUTPUT-
            keep : ResultList
                A ResultList object containing values from trajectories.
        """
        if filter_type == "clipped_sigmaG":
            filter_func = self.apply_clipped_sigmaG
        elif filter_type == "clipped_average":
            filter_func = self.apply_clipped_average
        elif filter_type == "kalman":
            filter_func = self.apply_kalman_filter

        keep = ResultList()
        likelihood_limit = False
        res_num = 0
        total_count = 0

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

            result_batch = ResultList()
            for i, trj in enumerate(results):
                # Stop as soon as we hit a result below our limit, because anything after
                # that is not guarrenteed to be valid due to potential on-GPU filtering.
                if trj.lh < lh_level:
                    likelihood_limit = True
                    break
                if trj.lh < max_lh:
                    psi_curve, phi_curve = search.lightcurve(trj)
                    row = ResultRow(trj, self._mjds)
                    row.set_psi_phi(psi_curve, phi_curve)
                    result_batch.append_result(row)
                    total_count += 1

            batch_size = result_batch.num_results()
            print("Extracted batch of %i results for total of %i" % (batch_size, total_count))
            if batch_size > 0:
                # Do rounds of filtering each is linear and we might be able to get
                # a speedup if we combine these.
                filter_func(result_batch, filter_params)
                result_batch.filter_on_num_valid_indices(3)
                result_batch.filter_on_likelihood(lh_level)

                # Add the results to the final set.
                keep.extend(result_batch)
            res_num += chunk_size
        return keep

    def get_coadd_stamps(self, result_idx, search, keep, radius=10, stamp_type="sum"):
        """
        Get the coadded stamps for the initial results from a kbmod search.

        Parameters
        ----------
        result_idx : The index of the result trajectories for which to compute
            the coadded stamps.
        search : `kbmod.search.Search`
            Search object.
        keep : ResultList
            The current set of results.
        radius : int
            The size of the stamp. Default 10 gives a 21x21 stamp.
            15 gives a 31x31 stamp, etc.
        stamp_type : string
            An input string to generate different kinds of stamps.
            'sum' or 'parallel_sum' - (default) A simple sum of all individual stamps
            'median' or 'cpp_median' - A per-pixel median of individual stamps.
            'mean' or 'cpp_median' - A per-pixel median of individual stamps.

        Returns
        -------
        coadd_stamps : list
            A list of numpy arrays containing the coadded stamp values for
            each trajectory.
        """
        start = time.time()

        # Create the list of trajectory results to use.
        trj_list = keep.trj_result_list(indices_to_use=result_idx)

        # Make the stamps.
        if stamp_type == "cpp_median" or stamp_type == "median":
            coadd_stamps = [np.array(stamp) for stamp in search.median_sci_stamps(trj_list, radius)]
        elif stamp_type == "cpp_mean":
            coadd_stamps = [np.array(stamp) for stamp in search.mean_sci_stamps(trj_list, radius)]
        elif stamp_type == "parallel_sum" or stamp_type == "sum":
            coadd_stamps = [np.array(stamp) for stamp in search.summed_sci_stamps(trj_list, radius)]
        else:
            coadd_stamps = []

        print(
            "Loaded {} coadded stamps. {:.3f}s elapsed".format(len(coadd_stamps), time.time() - start),
            flush=True,
        )
        return coadd_stamps

    def get_all_stamps(self, result_list, search, stamp_radius):
        """
        Get the stamps for the final results from a kbmod search.

        Parameters
        ----------
        result_list : ResultList
            The values from trajectories. The stamps get inserted into
            this data structure.
        search : kbmod.stack_search object
        stamp_radius : int
            The radius of the stamps to create.
        """
        stamp_edge = stamp_radius * 2 + 1
        for row in result_list.results:
            stamps = search.science_viz_stamps(row.trajectory, stamp_radius)
            row.all_stamps = np.array([np.array(stamp).reshape(stamp_edge, stamp_edge) for stamp in stamps])

    def apply_clipped_sigmaG(self, result_list, filter_params):
        """
        This function applies a clipped median filter to the results of a KBMOD
        search using sigmaG as a robust estimater of standard deviation.

        Parameters
        ----------
        result_list : ResultList
            The values from trajectories. This data gets modified directly
            by the filtering.
        filter_params : dictionary
            A dictionary of additional filtering parameters. Must
            contain sigmaG_filter_type.
        """
        print("Applying Clipped-sigmaG Filtering")
        self.lc_filter_type = filter_params["sigmaG_filter_type"]
        start_time = time.time()

        # Compute the coefficients for the filtering.
        if self.coeff is None:
            if self.sigmaG_lims is not None:
                self.percentiles = self.sigmaG_lims
            else:
                self.percentiles = [25, 75]
            self.coeff = self._find_sigmaG_coeff(self.percentiles)

        if self.num_cores > 1:
            zipped_curves = result_list.zipped_curve_list()

            keep_idx_results = []
            print("Starting pooling...")
            pool = mp.Pool(processes=self.num_cores)
            keep_idx_results = pool.starmap_async(self._clipped_sigmaG, zipped_curves)
            pool.close()
            pool.join()
            keep_idx_results = keep_idx_results.get()

            for i, res in enumerate(keep_idx_results):
                result_list.results[i].filter_indices(res[1])
        else:
            for i, row in enumerate(result_list.results):
                single_res = self._clipped_sigmaG(row.psi_curve, row.phi_curve, i)
                row.filter_indices(single_res[1])

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("{:.2f}s elapsed".format(time_elapsed))
        print("Completed filtering.", flush=True)
        print("---------------------------------------")

    def apply_clipped_average(self, result_list, filter_params):
        """
        This function applies a clipped median filter to the results of a KBMOD
        search.

        Parameters
        ----------
        result_list : ResultList
            The values from trajectories. This data gets modified directly
            by the filtering.
        filter_params : dictionary
            A dictionary of additional filtering parameters.
        """
        print("Applying Clipped-Average Filtering")
        start_time = time.time()

        if self.num_cores > 1:
            zipped_curves = result_list.zipped_curve_list()

            keep_idx_results = []
            print("Starting pooling...")
            pool = mp.Pool(processes=self.num_cores)
            keep_idx_results = pool.starmap_async(self._clipped_average, zipped_curves)
            pool.close()
            pool.join()
            keep_idx_results = keep_idx_results.get()

            for i, res in enumerate(keep_idx_results):
                result_list.results[i].filter_indices(res[1])
        else:
            for i, row in enumerate(result_list.results):
                single_res = self._clipped_average(row.psi_curve, row.phi_curve, i)
                row.filter_indices(single_res[1])

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("{:.2f}s elapsed".format(time_elapsed))
        print("Completed filtering.", flush=True)
        print("---------------------------------------")

    def _find_sigmaG_coeff(self, percentiles):
        z1 = percentiles[0] / 100
        z2 = percentiles[1] / 100

        x1 = self._invert_Gaussian_CDF(z1)
        x2 = self._invert_Gaussian_CDF(z2)
        coeff = 1 / (x2 - x1)
        print("sigmaG limits: [{},{}]".format(percentiles[0], percentiles[1]))
        print("sigmaG coeff: {:.4f}".format(coeff), flush=True)
        return coeff

    def _invert_Gaussian_CDF(self, z):
        if z < 0.5:
            sign = -1
        else:
            sign = 1
        x = sign * np.sqrt(2) * erfinv(sign * (2 * z - 1))  # mpmath.erfinv(sign * (2 * z - 1))
        return float(x)

    def _clipped_sigmaG(self, psi_curve, phi_curve, index, n_sigma=2):
        """
        This function applies a clipped median filter to a set of likelihood
        values. Points are eliminated if they are more than n_sigma*sigmaG away
        from the median.
        INPUT-
            psi_curve : numpy array
                A single Psi curve, likely a single row of a larger matrix of
                psi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['psi_curves'].
            phi_curve : numpy array
                A single Phi curve, likely a single row of a larger matrix of
                phi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['phi_curves'].
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            n_sigma : integer
                The number of standard deviations away from the median that
                the largest likelihood values (N=num_clipped) must be in order
                to be eliminated.
        OUTPUT-
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            good_index : numpy array
                The indices that pass the filtering for a given set of curves.
            new_lh : float
                The new maximum likelihood of the set of curves, after
                max_lh_index has been applied.
        """
        masked_phi = np.copy(phi_curve)
        masked_phi[masked_phi == 0] = 1e9
        if self.lc_filter_type == "lh":
            lh = psi_curve / np.sqrt(masked_phi)
            good_index = self._exclude_outliers(lh, n_sigma)
        elif self.lc_filter_type == "flux":
            flux = psi_curve / masked_phi
            good_index = self._exclude_outliers(flux, n_sigma)
        elif self.lc_filter_type == "both":
            lh = psi_curve / np.sqrt(masked_phi)
            good_index_lh = self._exclude_outliers(lh, n_sigma)
            flux = psi_curve / masked_phi
            good_index_flux = self._exclude_outliers(flux, n_sigma)
            good_index = np.intersect1d(good_index_lh, good_index_flux)
        else:
            print("Invalid filter type, defaulting to likelihood", flush=True)
            lh = psi_curve / np.sqrt(masked_phi)
            good_index = self._exclude_outliers(lh, n_sigma)

        if len(good_index) == 0:
            new_lh = 0
            good_index = [-1]
        else:
            new_lh = kb.calculate_likelihood_psi_phi(psi_curve[good_index], phi_curve[good_index])
        return (index, good_index, new_lh)

    def _exclude_outliers(self, lh, n_sigma):
        if self.clip_negative:
            lower_per, median, upper_per = np.percentile(
                lh[lh > 0], [self.percentiles[0], 50, self.percentiles[1]]
            )
            sigmaG = self.coeff * (upper_per - lower_per)
            nSigmaG = n_sigma * sigmaG
            good_index = np.where(
                np.logical_and(lh != 0, np.logical_and(lh > median - nSigmaG, lh < median + nSigmaG))
            )[0]
        else:
            lower_per, median, upper_per = np.percentile(lh, [self.percentiles[0], 50, self.percentiles[1]])
            sigmaG = self.coeff * (upper_per - lower_per)
            nSigmaG = n_sigma * sigmaG
            good_index = np.where(np.logical_and(lh > median - nSigmaG, lh < median + nSigmaG))[0]
        return good_index

    def _clipped_average(self, psi_curve, phi_curve, index, num_clipped=5, n_sigma=4, lower_lh_limit=-100):
        """
        This function applies a clipped median filter to a set of likelihood
        values. The largest likelihood values (N=num_clipped) are eliminated if
        they are more than n_sigma*standard deviation away from the median,
        which is calculated excluding the largest values.
        INPUT-
            psi_curve : numpy array
                A single Psi curve, likely a single row of a larger matrix of
                psi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['psi_curves'].
            phi_curve : numpy array
                A single Phi curve, likely a single row of a larger matrix of
                phi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['phi_curves'].
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            num_clipped : integer
                The number of likelihood values to consider eliminating. Only
                considers the largest N=num_clipped values.
            n_sigma : integer
                The number of standard deviations away from the median that
                the largest likelihood values (N=num_clipped) must be in order
                to be eliminated.
            lower_lh_limit : float
                Likelihood values lower than lower_lh_limit are automatically
                eliminated from consideration.
        OUTPUT-
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            max_lh_index : numpy array
                The indices that pass the filtering for a given set of curves.
            new_lh : float
                The new maximum likelihood of the set of curves, after
                max_lh_index has been applied.
        """
        max_lh_index = kb.clipped_ave_filtered_indices(
            psi_curve, phi_curve, num_clipped, n_sigma, lower_lh_limit
        )
        new_lh = -1.0
        if len(max_lh_index) > 0:
            new_lh = kb.calculate_likelihood_psi_phi(psi_curve[max_lh_index], phi_curve[max_lh_index])
        return (index, max_lh_index, new_lh)

    def apply_kalman_filter(self, result_list, filter_params):
        """
        This function applies a kalman filter to the results of a KBMOD search

        Parameters
        ----------
        result_list : ResultList
            The values from trajectories. This data gets modified directly
            by the filtering.
        filter_params : dictionary
            A dictionary of additional filtering parameters.
        """
        print("Applying Kalman Filtering")
        start_time = time.time()

        for row in result_list.results:
            keep_idx_results = kb.kalman_filtered_indices([row.psi_curve], [row.phi_curve])
            row.filter_indices(keep_idx_results[0][1])

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("{:.2f}s elapsed".format(time_elapsed))
        print("---------------------------------------")

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
        """
        This function filters result postage stamps based on their Gaussian
        Moments. Results with stamps that are similar to a Gaussian are kept.

        Parameters
        ----------
        result_list : ResultList
            The values from trajectories. This data gets modified directly
            by the filtering.
        search : kbmod.stack_search object
        center_thresh : float
            The fraction of the total flux that must be contained in a
            single central pixel.
        peak_offset : float array
            How far the brightest pixel in the stamp can be from the
            central pixel.
        mom_lims : float array
            The maximum limit of the xx, yy, xy, x, and y central moments
            of the stamp.
        chunk_size : int
            How many stamps to load and filter at a time.
        stamp_type : string
            Which method to use to generate stamps. One of median,
            cpp_median, mean, cpp_mean, or sum.
        stamp_radius : int
            The radius of the stamp.
        """
        # Set the stamp creation and filtering parameters.
        params = kb.stamp_parameters()
        params.radius = stamp_radius
        params.do_filtering = True
        params.center_thresh = center_thresh
        params.peak_offset_x = peak_offset[0]
        params.peak_offset_y = peak_offset[1]
        params.m20 = mom_lims[0]
        params.m02 = mom_lims[1]
        params.m11 = mom_lims[2]
        params.m10 = mom_lims[3]
        params.m01 = mom_lims[4]

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
        if result_list.num_results() > 0:
            print("Stamp filtering %i results" % result_list.num_results())
            while start_idx < result_list.num_results():
                end_idx = min([start_idx + chunk_size, result_list.num_results()])

                # Create a subslice of the results and the Boolean indices (as TrajectoryResults).
                # Note that the sum stamp type does not filter out lc_index.
                inds_to_use = [i for i in range(start_idx, end_idx)]
                results_slice = []
                if params.stamp_type != kb.StampType.STAMP_SUM:
                    results_slice = result_list.trj_result_list(indices_to_use=inds_to_use)
                else:
                    trj_list = result_list.trajectory_list(indices_to_use=inds_to_use)
                    results_slice = [kb.trj_result(x, num_times) for x in trj_list]

                # Create and filter the results.
                stamps_slice = search.gpu_coadded_stamps(results_slice, params)
                for ind, stamp in enumerate(stamps_slice):
                    if stamp.get_width() > 1:
                        result_list.results[ind + start_idx].stamp = np.array(stamp)
                        all_valid_inds.append(ind + start_idx)

                # Move to the next chunk.
                start_idx += chunk_size
                del stamps_slice

        # Do the actual filtering of results
        result_list.filter_results(all_valid_inds)
        print("Keeping %i results" % result_list.num_results(), flush=True)

        end_time = time.time()
        time_elapsed = end_time - start_time
        print("{:.2f}s elapsed".format(time_elapsed))

    def apply_clustering(self, result_list, cluster_params):
        """
        This function clusters results that have similar trajectories.

        Parameters
        ----------
        result_list : ResultList
            The values from trajectories. This data gets modified directly
            by the filtering.
        cluster_params : dictionary
            Contains values concerning the image and search initial settings
            including: x_size, y_size, vel_lims, ang_lims, and mjd.
        """
        # Skip clustering if there is nothing to cluster.
        if result_list.num_results() == 0:
            return
        print("Clustering %i results" % result_list.num_results(), flush=True)

        # Do the clustering and the filtering.
        cluster_idx = self._cluster_results(
            np.array(result_list.trajectory_list()),
            cluster_params["x_size"],
            cluster_params["y_size"],
            cluster_params["vel_lims"],
            cluster_params["ang_lims"],
            cluster_params["mjd"],
        )
        result_list.filter_results(cluster_idx)
        del cluster_idx

    def _cluster_results(self, results, x_size, y_size, v_lim, ang_lim, mjd_times, cluster_args=None):
        """
        This function clusters results and selects the highest-likelihood
        trajectory from a given cluster.

        Parameters
        ----------
        results : list
            A list of kbmod trajectories
        x_size : list
            The width of the images used in the kbmod stack, such as are
            stored in image_params['x_size'].
        y_size : list
            The height of the images used in the kbmod stack, such as are
            stored in image_params['y_size'].
        v_lim : list
            The velocity limits of the search, such as are stored in
            image_params['v_lim'].
        ang_lim : list
            The angle limits of the search, such as are stored in
            image_params['ang_lim']
        cluster_args : dictionary
            Arguments to pass to dbscan or OPTICS.

        Returns
        -------
        top_vals : numpy array
            An array of the indices for the best trajectories of each
            individual cluster.
        """
        if self.cluster_function == "DBSCAN":
            default_cluster_args = dict(eps=self.eps, min_samples=1, n_jobs=-1)
        elif self.cluster_function == "OPTICS":
            default_cluster_args = dict(max_eps=self.eps, min_samples=2, n_jobs=-1)

        if cluster_args is not None:
            default_cluster_args.update(cluster_args)
        cluster_args = default_cluster_args

        x_arr = []
        y_arr = []
        vx_arr = []
        vy_arr = []
        vel_arr = []
        ang_arr = []
        times = mjd_times - mjd_times[0]

        for line in results:
            x_arr.append(line.x)
            y_arr.append(line.y)
            vx_arr.append(line.x_v)
            vy_arr.append(line.y_v)
            vel_arr.append(np.sqrt(line.x_v**2.0 + line.y_v**2.0))
            ang_arr.append(np.arctan2(line.y_v, line.x_v))

        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        vx_arr = np.array(vx_arr)
        vy_arr = np.array(vy_arr)
        vel_arr = np.array(vel_arr)
        ang_arr = np.array(ang_arr)

        scaled_x = x_arr / x_size
        scaled_y = y_arr / y_size
        scaled_vel = (vel_arr - v_lim[0]) / (v_lim[1] - v_lim[0])
        scaled_ang = (ang_arr - ang_lim[0]) / (ang_lim[1] - ang_lim[0])

        if self.cluster_function == "DBSCAN":
            cluster = DBSCAN(**cluster_args)
        elif self.cluster_function == "OPTICS":
            cluster = OPTICS(**cluster_args)

        if self.cluster_type == "all":
            cluster.fit(np.array([scaled_x, scaled_y, scaled_vel, scaled_ang], dtype=float).T)
        elif self.cluster_type == "position":
            cluster.fit(np.array([scaled_x, scaled_y], dtype=float).T)
        elif self.cluster_type == "mid_position":
            median_time = np.median(times)
            mid_x_arr = x_arr + median_time * vx_arr
            mid_y_arr = y_arr + median_time * vy_arr
            scaled_mid_x = mid_x_arr / x_size
            scaled_mid_y = mid_y_arr / y_size
            cluster.fit(np.array([scaled_mid_x, scaled_mid_y], dtype=float).T)

        top_vals = []
        for cluster_num in np.unique(cluster.labels_):
            cluster_vals = np.where(cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])

        del cluster

        return top_vals
