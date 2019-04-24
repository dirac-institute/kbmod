import os
import shutil
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import csv
import pdb
import astropy.coordinates as astroCoords
import astropy.units as u
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from skimage import measure
from collections import OrderedDict

class SharedTools():
    def __init__(self):

        return

    def generate_results_dict(self):
        """
        Return an empty results dictionary. All values needed for a results
        dictionary should be added here.
        """
        keep = {'stamps': [], 'new_lh': [], 'results': [], 'times': [],
                'lc': [], 'lc_index':[], 'all_stamps':[], 'psi_curves':[],
                'phi_curves':[], 'final_results': []}
        return(keep)

class Interface(SharedTools):

    def __init__(self):

        return


    def return_filename(self, visit_num):

        hits_file = '%i.fits' % visit_num

        return hits_file

    def get_folder_visits(self, folder_visits):

        patch_visit_ids = np.array([int(visit_name[0:6]) for visit_name in folder_visits])

        return patch_visit_ids

    def load_images(self, im_filepath, time_file, mjd_lims=None):
        """
        This function loads images and ingests them into a search object
        Input-
            im_filepath : string
                Image file path from which to load images
            time_file : string
                File name containing image times
        Output-
            search : kbmod.stack_search object
            image_params : dictionary
                Contains image parameters such as ecliptic angle and mean Julian day
        """
        # Empty for now. Will contain x_size, y_size, ec_angle, and mjd before being returned.
        image_params = {}

        print('---------------------------------------')
        print("Loading Images")
        print('---------------------------------------')
        visit_nums, visit_times = np.genfromtxt(time_file, unpack=True)
        image_time_dict = OrderedDict()
        for visit_num, visit_time in zip(visit_nums, visit_times):
            image_time_dict[str(int(visit_num))] = visit_time

        patch_visits = sorted(os.listdir(im_filepath))
        patch_visit_ids = self.get_folder_visits(patch_visits)
        patch_visit_times = np.array([image_time_dict[str(visit_id)]
                                      for visit_id in patch_visit_ids])

        if mjd_lims is None:
            use_images = patch_visit_ids
        else:
            visit_only = np.where(((patch_visit_times > mjd_lims[0])
                                   & (patch_visit_times < mjd_lims[1])))[0]
            print(visit_only)
            use_images = patch_visit_ids[visit_only]

        image_params['mjd'] = np.array([image_time_dict[str(visit_id)]
                                        for visit_id in use_images])
        times = image_params['mjd'] - image_params['mjd'][0]
        file_path = '%s/%s' % (im_filepath, self.return_filename(use_images[0]))
        hdulist = fits.open(file_path)
        wcs = WCS(hdulist[1].header)
        image_params['ec_angle'] = self._calc_ecliptic_angle(wcs)
        del(hdulist)

        images = [kb.layered_image(
            '%s/%s' % (im_filepath, self.return_filename(f)))
            for f in np.sort(use_images)]

        print('Loaded %i images' %(len(images)))
        stack = kb.image_stack(images)

        stack.set_times(times)
        print("Times set")

        image_params['x_size'] = stack.get_width()
        image_params['y_size'] = stack.get_width()
        image_params['times']  = stack.get_times()
        return(stack,image_params)

    def load_results(self,search,image_params,res_filepath,lh_level):
        """
        Load results that are output by the gpu grid search
        """

        keep = self.generate_results_dict() 

        likelihood_limit = False
        res_num = 0
        chunk_size = 500000
        psi_curves = None
        phi_curves = None
        all_results = []
        print('---------------------------------------')
        print("Retrieving Results")
        print('---------------------------------------')
        while likelihood_limit is False:
            print('Getting results...')
            results = search.get_results(res_num,chunk_size)
            chunk_headers = ("Chunk Start", "Chunk Max Likelihood",
                             "Chunk Min. Likelihood")
            chunk_values = (res_num, results[0].lh, results[-1].lh)
            for header, val, in zip(chunk_headers, chunk_values):
                if type(val) == np.int:
                    print('%s = %i' % (header, val))
                else:
                    print('%s = %.2f' % (header, val))
            print('---------------------------------------')
            # Find the size of the psi phi curves and preallocate arrays
            foo_psi,_=search.lightcurve(results[0])
            curve_len = len(foo_psi.flatten())
            curve_shape = [len(results),curve_len]
            prealloc_matrix = np.zeros(curve_shape)
            if (psi_curves is None) or (phi_curves is None):
                psi_curves = np.copy(prealloc_matrix)
                phi_curves = np.copy(prealloc_matrix)
            else:
                psi_curves = np.append(psi_curves, prealloc_matrix, axis=0)
                phi_curves = np.append(phi_curves, prealloc_matrix, axis=0)

            for i,line in enumerate(results):
                #pdb.set_trace()
                curve_index = i+res_num
                psi_curve, phi_curve = search.lightcurve(line)
                psi_curves[curve_index] = psi_curve
                phi_curves[curve_index] = phi_curve
                if line.lh < lh_level:
                    likelihood_limit = True
                    break
            all_results.append(results)
            res_num+=chunk_size

        # Trim phi and psi to eliminate excess preallocated arrays
        psi_curves = psi_curves[0:curve_index]
        phi_curves = phi_curves[0:curve_index]
        keep['psi_curves'] = psi_curves
        keep['phi_curves'] = phi_curves
        keep['results'] = np.concatenate(all_results)[0:curve_index]
        pdb.set_trace()
        print('Retrieved %i results' %(np.shape(psi_curves)[0]))
        return(keep)

    def save_results(self, res_filepath, out_suffix, keep):
        """
        Save results from a given search method
        (either region search or grid search)
        Input-
            res_filepath : string
            out_suffix : string
                Suffix to append to the output file name
            keep : dictionary
                Dictionary that contains the values to keep and print to filtering
        """

        print('---------------------------------------')
        print("Saving Results")
        print('---------------------------------------')
        np.savetxt('%s/results_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep['results'])[keep['final_results']], fmt='%s')
        with open('%s/lc_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['lc'])[keep['final_results']])
        with open('%s/psi_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['psi_curves'])[keep['final_results']])
        with open('%s/phi_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['phi_curves'])[keep['final_results']])
        with open('%s/lc_index_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['lc_index'])[keep['final_results']])
        with open('%s/times_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['times'])[keep['final_results']])
        np.savetxt(
            '%s/filtered_likes_%s.txt' % (res_filepath, out_suffix),
            np.array(keep['new_lh'])[keep['final_results']], fmt='%.4f')
        np.savetxt(
            '%s/ps_%s.txt' % (res_filepath, out_suffix),
            np.array(keep['stamps']).reshape(
                len(keep['stamps']), 441)[keep['final_results']], fmt='%.4f')
        stamps_to_save = np.array(keep['all_stamps'])[keep['final_results']]
        np.save('%s/all_ps_%s.npy' % (res_filepath, out_suffix), stamps_to_save)

    def _calc_ecliptic_angle(self, test_wcs, angle_to_ecliptic=0.):

        wcs = [test_wcs]
        pixel_coords = [[],[]]
        pixel_start = [[1000, 2000]]
        angle = np.float(angle_to_ecliptic)
        vel_array = np.array([[6.*np.cos(angle), 6.*np.sin(angle)]])
        time_array = [0.0, 1.0, 2.0]

        vel_par_arr = vel_array[:, 0]
        vel_perp_arr = vel_array[:, 1]

        if type(vel_par_arr) is not np.ndarray:
            vel_par_arr = [vel_par_arr]
        if type(vel_perp_arr) is not np.ndarray:
            vel_perp_arr = [vel_perp_arr]
        for start_loc, vel_par, vel_perp in zip(pixel_start,
                                                vel_par_arr, vel_perp_arr):

            start_coord = astroCoords.SkyCoord.from_pixel(start_loc[0],
                                                          start_loc[1],
                                                          wcs[0])
            eclip_coord = start_coord.geocentrictrueecliptic
            eclip_l = []
            eclip_b = []
            for time_step in time_array:
                eclip_l.append(eclip_coord.lon + vel_par*time_step*u.arcsec)
                eclip_b.append(eclip_coord.lat + vel_perp*time_step*u.arcsec)
            eclip_vector = astroCoords.SkyCoord(eclip_l, eclip_b,
                                                frame='geocentrictrueecliptic')
            pixel_coords_set = astroCoords.SkyCoord.to_pixel(eclip_vector, wcs[0])
            pixel_coords[0].append(pixel_coords_set[0])
            pixel_coords[1].append(pixel_coords_set[1])

        pixel_coords = np.array(pixel_coords)
        x_dist = pixel_coords[0, 0, -1] - pixel_coords[0, 0, 0]
        y_dist = pixel_coords[1, 0, -1] - pixel_coords[1, 0, 0]

        eclip_angle = np.arctan(y_dist/x_dist)

        return eclip_angle

class PostProcess(SharedTools):

    def __init__(self):
        return

    def apply_mask(self,stack, mask_num_images=2,mask_threshold=120.):

        flags = ~0 # mask pixels with any flags
        flag_exceptions = [32,39] # unless it has one of these special combinations of flags
        master_flags = int('100111', 2) # mask any pixels which have any of
        # these flags in more than two images

        # Apply masks
        stack.apply_mask_flags(flags, flag_exceptions)
        stack.apply_master_mask(master_flags, mask_num_images)

        stack.grow_mask()
        stack.grow_mask()
        
        # This applies a mask to pixels with more than 120 counts
        stack.apply_mask_threshold(mask_threshold)
        return(stack)

    def apply_kalman_filter(self, old_results, search, image_params, lh_level):
        """
        Apply a kalman filter to the results of a KBMOD search
            Input-
                keep : dictionary
                    Dictionary containing values from trajectories. When input,
                    it should have at least 'psi_curves', 'phi_curves', and
                    'results'. These are populated in Interface.load_results().
                search : kbmod.stack_search object
                image_params : dictionary
                    Dictionary containing parameters about the images that were
                    searched over. apply_kalman_filter only uses MJD
                lh_level : float
                    Minimum likelihood to search
            Output-
                keep : dictionary
                    Dictionary containing values from trajectories that pass
                    the kalman filtering.
        """
        print('---------------------------------------')
        print("Applying Kalman Filter to Results")
        print('---------------------------------------')
        # Make copies of the values in 'old_results' and create a new dict
        psi_curves = np.copy(old_results['psi_curves'])
        phi_curves = np.copy(old_results['phi_curves'])
        masked_phi_curves = np.copy(phi_curves)
        masked_phi_curves[masked_phi_curves==0] = 1e9
        results = old_results['results']
        keep = self.generate_results_dict()

        print('Starting pooling...')
        #pdb.set_trace()
        #foo=[]
       # for i in range(len(psi_curves)):
        #    foo.append(self._return_indices(psi_curves[i],phi_curves[i],i))
        pool = mp.Pool(processes=16)
        zipped_curves = zip(
            psi_curves, phi_curves, [j for j in range(len(psi_curves))])
        keep_idx_results = pool.starmap_async(
            self._return_indices, zipped_curves)
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()

        if len(keep_idx_results[0]) < 3:
            keep_idx_results = [(0, [-1], 0.)]
        for result_on in range(len(psi_curves)):
            if keep_idx_results[result_on][1][0] == -1:
                continue
            elif len(keep_idx_results[result_on][1]) < 3:
                continue
            elif keep_idx_results[result_on][2] < lh_level:
                continue
            else:
                keep_idx = keep_idx_results[result_on][1]
                new_likelihood = keep_idx_results[result_on][2]
                keep['results'].append(results[result_on])
                keep['new_lh'].append(new_likelihood)
                stamps = search.sci_stamps(results[result_on], 10)
                all_stamps = np.array([np.array(stamp).reshape(21,21) for stamp in stamps])
                stamp_arr = np.array([np.array(stamps[s_idx]) for s_idx in keep_idx])
                keep['all_stamps'].append(all_stamps)
                keep['stamps'].append(np.sum(stamp_arr, axis=0))
                keep['lc'].append((psi_curves[result_on]/masked_phi_curves[result_on]))
                keep['lc_index'].append(keep_idx)
                keep['psi_curves'].append(psi_curves[result_on])
                keep['phi_curves'].append(phi_curves[result_on])
                keep['times'].append(image_params['mjd'][keep_idx])
        print('Kalman filtering keeps %i results' %(len(keep['results'])))
        return(keep)

    def apply_stamp_filter(self, keep):
        """
        Filter result postage stamps based on their Gaussian Moments 
        Input-
            keep : dictionary
                Contains the values of which results were kept from the search
                algorithm
            image_params : dictionary
                Contains values concerning the image and search initial
                settings
        Output-
            keep : dictionary
                Contains the values of which results were kept from the search
                algorithm
        """
        lh_sorted_idx = np.argsort(np.array(keep['new_lh']))[::-1]
        pdb.set_trace()
        print(len(lh_sorted_idx))
        if len(lh_sorted_idx) > 0:
            print("Stamp filtering %i results" % len(lh_sorted_idx))
            pool = mp.Pool(processes=16)
            stamp_filt_pool = pool.map_async(self._stamp_filter_parallel,
                                             np.array(keep['stamps'])[lh_sorted_idx])
            pool.close()
            pool.join()
            stamp_filt_results = stamp_filt_pool.get()
            stamp_filt_idx = lh_sorted_idx[np.where(np.array(stamp_filt_results) == 1)]
            if len(stamp_filt_idx) > 0:
                keep['final_results'] = stamp_filt_idx
            else:
                keep['final_results'] = []
            del(stamp_filt_results)
            del(stamp_filt_idx)
            del(stamp_filt_pool)
        else:
            keep['final_results'] = lh_sorted_idx
        print('Keeping %i results' % len(keep['final_results']))
        return(keep)

    def apply_clustering(self, keep, image_params):
        """
        Cluster results that have similar trajectories
        Input-
            keep : Dictionary
                Contains the values of which results were kept from the search
                algorithm
            image_params : dictionary
                Contains values concerning the image and search initial
                settings
        Output-
            keep : Dictionary
                Contains the values of which results were kept from the search
                algorithm
        """
        results_indices = keep['final_results']
        print("Clustering %i results" % len(results_indices))
        if len(keep['final_results'])>0:
            cluster_idx = self._cluster_results(
                np.array(keep['results'])[results_indices], 
                image_params['x_size'], image_params['y_size'],
                image_params['vel_lims'], image_params['ang_lims'])
            keep['final_results'] = keep['final_results'][cluster_idx]
            del(cluster_idx)
        print('Keeping %i results' % len(keep['final_results']))
        return(keep)

    def _kalman_filter(self, obs, var):

        xhat = np.zeros(len(obs))
        P = np.zeros(len(obs))
        xhatminus = np.zeros(len(obs))
        Pminus = np.zeros(len(obs))
        K = np.zeros(len(obs))

        Q = 1.
        R = np.copy(var)

        xhat[0] = obs[0]
        P[0] = R[0]

        for k in range(1,len(obs)):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q

            K[k] = Pminus[k] / (Pminus[k] + R[k])
            xhat[k] = xhatminus[k] + K[k]*(obs[k]-xhatminus[k])
            P[k] = (1-K[k])*Pminus[k]
        return xhat, P

    def _return_indices(self, psi_values, phi_values, val_on):
        masked_phi_values = np.copy(phi_values)
        masked_phi_values[masked_phi_values==0] = 1e9
        flux_vals = psi_values/masked_phi_values
        flux_idx = np.where(flux_vals != 0.)[0]
        if len(flux_idx) < 2:
            return ([], [-1], [])
        fluxes = flux_vals[flux_idx]
        inv_flux = np.array(masked_phi_values[flux_idx])
        inv_flux[inv_flux < -999.] = 9999999.
        f_var = (1./inv_flux)

        ## 1st pass
        #f_var = #var_curve[flux_idx]#np.var(fluxes)*np.ones(len(fluxes))
        kalman_flux, kalman_error = self._kalman_filter(fluxes, f_var)
        if np.min(kalman_error) < 0.:
            return ([], [-1], [])
        deviations = np.abs(kalman_flux - fluxes) / kalman_error**.5

        #print(deviations, fluxes)
        # keep_idx = np.where(deviations < 500.)[0]
        keep_idx = np.where(deviations < 5.)[0]

        ## Second Pass (reverse order in case bright object is first datapoint)
        kalman_flux, kalman_error = self._kalman_filter(fluxes[::-1], f_var[::-1])
        if np.min(kalman_error) < 0.:
            return ([], [-1], [])
        deviations = np.abs(kalman_flux - fluxes[::-1]) / kalman_error**.5
        #print(fluxes, f_var, kalman_flux, kalman_error**.5, deviations)
        # keep_idx_back = np.where(deviations < 500.)[0]
        keep_idx_back = np.where(deviations < 5.)[0]

        if len(keep_idx) >= len(keep_idx_back):
            new_psi = psi_values[flux_idx[keep_idx]]
            new_phi = phi_values[flux_idx[keep_idx]]
            new_lh = self._compute_lh(new_psi,new_phi)
            return (val_on, flux_idx[keep_idx], new_lh)
        else:
            reverse_idx = len(flux_idx)-1 - keep_idx_back
            new_psi = psi_values[flux_idx[reverse_idx]]
            new_phi = phi_values[flux_idx[reverse_idx]]
            new_lh = self._compute_lh(new_psi,new_phi)
            return (val_on, flux_idx[reverse_idx], new_lh)

    def _compute_lh(self, psi_values, phi_values):
        lh = np.sum(psi_values)/np.sqrt(np.sum(phi_values))
        return(lh)

    def _cluster_results(self, results, x_size, y_size,
                        v_lim, ang_lim, dbscan_args=None):

        default_dbscan_args = dict(eps=0.03, min_samples=-1, n_jobs=-1)

        if dbscan_args is not None:
            default_dbscan_args.update(dbscan_args)
        dbscan_args = default_dbscan_args

        x_arr = []
        y_arr = []
        vel_arr = []
        ang_arr = []

        for line in results:
            x_arr.append(line.x)
            y_arr.append(line.y)
            vel_arr.append(np.sqrt(line.x_v**2. + line.y_v**2.))
            ang_arr.append(np.arctan(line.y_v/line.x_v))

        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        vel_arr = np.array(vel_arr)
        ang_arr = np.array(ang_arr)

        scaled_x = x_arr/x_size
        scaled_y = y_arr/y_size
        scaled_vel = (vel_arr - v_lim[0])/(v_lim[1] - v_lim[0])
        scaled_ang = (ang_arr - ang_lim[0])/(ang_lim[1] - ang_lim[0])

        db_cluster = DBSCAN(**dbscan_args)

        db_cluster.fit(np.array([scaled_x, scaled_y,
                                scaled_vel, scaled_ang], dtype=np.float).T)

        top_vals = []
        for cluster_num in np.unique(db_cluster.labels_):
            cluster_vals = np.where(db_cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])

        del(db_cluster)

        return top_vals
    
    def _stamp_filter_parallel(self,stamps):

        center_thresh = 0.03

        s = stamps - np.min(stamps)
        s /= np.sum(s)
        s = np.array(s, dtype=np.dtype('float64')).reshape(21, 21)
        mom = measure.moments_central(s, center=(10,10))
        mom_list = [mom[2, 0], mom[0, 2], mom[1, 1], mom[1, 0], mom[0, 1]]
        peak_1, peak_2 = np.where(s == np.max(s))

        if len(peak_1) > 1:
            peak_1 = np.max(np.abs(peak_1-10.))

        if len(peak_2) > 1:
            peak_2 = np.max(np.abs(peak_2-10.))

        if ((mom_list[0] < 35.5) & (mom_list[1] < 35.5) &
                (np.abs(mom_list[2]) < 1.) &
                (np.abs(mom_list[3]) < .25) & (np.abs(mom_list[4]) < .25) &
                (np.abs(peak_1 - 10.) < 2.) & (np.abs(peak_2 - 10.) < 2.)):
            if np.max(stamps/np.sum(stamps)) > center_thresh:
                keep_stamps = 1
            else:
                keep_stamps = 0
        else:
            keep_stamps = 0

        return keep_stamps

