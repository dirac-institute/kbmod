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

from .file_utils import *
from .image_info import *
from .result_list import *
import kbmod.search as kb


class Interface(SharedTools):
    """This class manages the KBMOD interface with the local filesystem, the cpp
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
        default_psf,
        verbose=False,
    ):
        """This function loads images and ingests them into a search object.

        Parameters
        ----------
        im_filepath : string
            Image file path from which to load images.
        time_file : string
            File name containing image times.
        psf_file : string
            File name containing the image-specific PSFs.
            If set to None the code will use the provided default psf for
            all images.
        mjd_lims : list of ints
            Optional MJD limits on the images to search.
        default_psf : `psf`
            The default PSF in case no image-specific PSF is provided.
        verbose : bool
            Use verbose output (mainly for debugging).

        Returns
        -------
            stack : `kbmod.image_stack`
                The stack of images loaded.
            img_info : `ImageInfo`
                The information for the images loaded.
        """
        print("---------------------------------------")
        print("Loading Images")
        print("---------------------------------------")

        # Load a mapping from visit numbers to the visit times. This dictionary stays
        # empty if no time file is specified.
        image_time_dict = FileUtils.load_time_dictionary(time_file)
        if verbose:
            print(f"Loaded {len(image_time_dict)} time stamps.")

        # Load a mapping from visit numbers to PSFs. This dictionary stays
        # empty if no time file is specified.
        image_psf_dict = FileUtils.load_psf_dictionary(psf_file)
        if verbose:
            print(f"Loaded {len(image_psf_dict)} image PSFs stamps.")

        # Retrieve the list of visits (file names) in the data directory.
        patch_visits = sorted(os.listdir(im_filepath))

        # Load the images themselves.
        img_info = ImageInfoSet()
        images = []
        visit_times = []
        for visit_file in np.sort(patch_visits):
            # Skip non-fits files.
            if not ".fits" in visit_file:
                if verbose:
                    print(f"Skipping non-FITS file {visit_file}")
                continue

            # Compute the full file path for loading.
            full_file_path = os.path.join(im_filepath, visit_file)

            # Load the image info from the FITS header.
            header_info = ImageInfo()
            header_info.populate_from_fits_file(full_file_path)

            # Skip files without a valid visit ID.
            if header_info.visit_id is None:
                if verbose:
                    print(f"WARNING: Unable to extract visit ID for {visit_file}.")
                continue

            # Compute the time stamp as a MJD float. If there is an entry in the
            # timestamp file, defer to that. Otherwise use the value from the header.
            time_stamp = -1.0
            if header_info.visit_id in image_time_dict:
                time_stamp = image_time_dict[header_info.visit_id]
            else:
                time_obj = header_info.get_epoch(none_if_unset=True)
                if time_obj is not None:
                    time_stamp = time_obj.mjd

            if time_stamp <= 0.0:
                if verbose:
                    print(f"WARNING: No valid timestamp provided for {visit_file}.")
                continue

            # Check if we should filter the record based on the time bounds.
            if mjd_lims is not None and (time_stamp < mjd_lims[0] or time_stamp > mjd_lims[1]):
                if verbose:
                    print(f"Pruning file {visit_file} by timestamp={time_stamp}.")
                continue

            # Check if the image has a specific PSF.
            psf = default_psf
            if header_info.visit_id in image_psf_dict:
                psf = kb.psf(image_psf_dict[header_info.visit_id])

            # Load the image file and set its time.
            if verbose:
                print(f"Loading file: {full_file_path}")
            img = kb.layered_image(full_file_path, psf)
            img.set_time(time_stamp)

            # Save the file, time, and image information.
            img_info.append(header_info)
            visit_times.append(time_stamp)
            images.append(img)

        print(f"Loaded {len(images)} images")
        stack = kb.image_stack(images)

        # Create a list of visit times and visit times shifted to 0.0.
        img_info.set_times_mjd(np.array(visit_times))
        times = img_info.get_zero_shifted_times()
        stack.set_times(times)
        print("Times set", flush=True)

        return (stack, img_info)

    def save_results(self, res_filepath, out_suffix, keep, all_times):
        """This function saves results from a given search method.

        Parameters
        ----------
        res_filepath : string
            The filepath for the results.
        out_suffix : string
            Suffix to append to the output file name
        keep : `ResultList`
            ResultList object containing the values to keep and print to file.
        all_times : list
            A list of times.
        """
        print("---------------------------------------")
        print("Saving Results")
        print("---------------------------------------", flush=True)
        keep.save_to_files(res_filepath, out_suffix)


class PostProcess(SharedTools):
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
        self.mask_bits_dict = config["mask_bits_dict"]
        self.flag_keys = config["flag_keys"]
        self.repeated_flag_keys = config["repeated_flag_keys"]
        self._mjds = mjds

    def apply_mask(self, stack, mask_num_images=2, mask_threshold=None, mask_grow=10):
        """This function applys a mask to the images in a KBMOD stack. This mask
        sets a high variance for masked pixels

        Parameters
        ----------
        stack : `kbmod.image_stack`
            The stack before the masks have been applied.
        mask_num_images : int
            The minimum number of images in which a masked pixel must appear in
            order for it to be masked out. E.g. if masked_num_images = 2, then an
            object must appear in the same place in at least two images in order
            for the variance at that location to be increased.
        mask_threshold : float
            Any pixel with a flux greater than mask_threshold is masked out.
        mask_grow : int
            The number of pixels by which to grow the mask.

        Returns
        -------
        stack : `kbmod.image_stack`
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
                    psi_curve = np.array(search.psi_curves(trj))
                    phi_curve = np.array(search.phi_curves(trj))
                    row.set_psi_phi(psi_curve, phi_curve)
                    result_batch.append_result(row)
                    total_count += 1

            batch_size = result_batch.num_results()
            print("Extracted batch of %i results for total of %i" % (batch_size, total_count))
            if batch_size > 0:
                self.apply_clipped_sigmaG(result_batch)
                result_batch.filter_on_stats(lh_level, 3)

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
        search : `kbmod.stack_search`
            The search object
        stamp_radius : int
            The radius of the stamps to create.
        """
        stamp_edge = stamp_radius * 2 + 1
        for row in result_list.results:
            stamps = search.science_viz_stamps(row.trajectory, stamp_radius)
            row.all_stamps = np.array([np.array(stamp).reshape(stamp_edge, stamp_edge) for stamp in stamps])

    def apply_clipped_sigmaG(self, result_list):
        """This function applies a clipped median filter to the results of a KBMOD
        search using sigmaG as a robust estimater of standard deviation.

        Parameters
        ----------
        result_list : `ResultList`
            The values from trajectories. This data gets modified directly
            by the filtering.
        """
        print("Applying Clipped-sigmaG Filtering")
        start_time = time.time()

        # Compute the coefficients for the filtering.
        if self.coeff is None:
            if self.sigmaG_lims is not None:
                self.percentiles = self.sigmaG_lims
            else:
                self.percentiles = [25, 75]
            self.coeff = self._find_sigmaG_coeff(self.percentiles)

        if self.num_cores > 1:
            zipped_curves = result_list.zip_phi_psi_idx()

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
        """This function applies a clipped median filter to a set of likelihood
        values. Points are eliminated if they are more than n_sigma*sigmaG away
        from the median.

        Parameters
        ----------
        psi_curve : numpy array
            A single Psi curve, likely from a `ResultRow`.
        phi_curve : numpy array
            A single Phi curve, likely from a `ResultRow`.
        index : int
            The index of the ResultRow being processed. Used track
            multiprocessing.
        n_sigma : int
            The number of standard deviations away from the median that
            the largest likelihood values (N=num_clipped) must be in order
            to be eliminated.

        Returns
        -------
        index : int
            The index of the ResultRow being processed. Used track multiprocessing.
        good_index: numpy array
            The indices that pass the filtering for a given set of curves.
        new_lh : float
            The new maximum likelihood of the set of curves, after max_lh_index has
            been applied.
        """
        masked_phi = np.copy(phi_curve)
        masked_phi[masked_phi == 0] = 1e9

        lh = psi_curve / np.sqrt(masked_phi)
        good_index = self._exclude_outliers(lh, n_sigma)
        if len(good_index) == 0:
            new_lh = 0
            good_index = []
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
        search : `kbmod.stack_search`
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
        if result_list.num_results() <= 0:
            print("Skipping. Nothing to filter.")
            return

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
        cluster_idx = self._cluster_results(
            np.array(result_list.trajectory_list()),
            cluster_params["x_size"],
            cluster_params["y_size"],
            cluster_params["vel_lims"],
            cluster_params["ang_lims"],
            cluster_params["mjd"],
        )
        result_list.filter_results(cluster_idx)

    def _cluster_results(self, results, x_size, y_size, v_lim, ang_lim, mjd_times, cluster_args=None):
        """This function clusters results and selects the highest-likelihood
        trajectory from a given cluster.

        Parameters
        ----------
        results : list
            A list of kbmod trajectory results.
        x_size : int
            The width of the images (in pixels) used in the kbmod stack, such
            as are stored in image_params['x_size'].
        y_size : int
            The height of the images (in pixels) used in the kbmod stack such
            as are stored in image_params['y_size'].
        v_lim : list
            The velocity limits of the search, such as are stored in
            image_params['v_lim']. The first two elements are used and represent
            the minimum (v_lim[0]) and maximum (v_lim[1]) velocities used in the
            search.
        ang_lim : list
            The angle limits of the search, such as are stored in image_params['ang_lim'].
            The first two elements are used and represent the minimum (ang_lim[0]) and
            maximum (ang_lim[1]) angles used in the search.
        cluster_args : dict
            Arguments to pass to dbscan or OPTICS.

        Returns
        -------
        top_vals : numpy array
            An array of the indices for the best trajectories of each individual cluster.
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

        v_scale = (v_lim[1] - v_lim[0]) if v_lim[1] != v_lim[0] else 1.0
        scaled_vel = (vel_arr - v_lim[0]) / v_scale

        a_scale = (ang_lim[1] - ang_lim[0]) if ang_lim[1] != ang_lim[0] else 1.0
        scaled_ang = (ang_arr - ang_lim[0]) / a_scale

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

        return top_vals
