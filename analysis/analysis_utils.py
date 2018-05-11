import os
import shutil
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import astropy.coordinates as astroCoords
import astropy.units as u
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from skimage import measure
from collections import OrderedDict


def kalman_filter(obs, var):

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


def return_indices(psi_values, phi_values, val_on):

    flux_vals = psi_values/phi_values
    flux_idx = np.where(flux_vals != 0.)[0]
    if len(flux_idx) < 2:
        return ([], [-1], [])
    fluxes = flux_vals[flux_idx]
    inv_flux = np.array(phi_values[flux_idx])
    inv_flux[inv_flux < -999.] = 9999999.
    f_var = (1./inv_flux)

    ## 1st pass
    #f_var = #var_curve[flux_idx]#np.var(fluxes)*np.ones(len(fluxes))
    kalman_flux, kalman_error = kalman_filter(fluxes, f_var)
    if np.min(kalman_error) < 0.:
        return ([], [-1], [])
    deviations = np.abs(kalman_flux - fluxes) / kalman_error**.5

    #print(deviations, fluxes)
    # keep_idx = np.where(deviations < 500.)[0]
    keep_idx = np.where(deviations < 5.)[0]

    ## Second Pass (reverse order in case bright object is first datapoint)
    kalman_flux, kalman_error = kalman_filter(fluxes[::-1], f_var[::-1])
    if np.min(kalman_error) < 0.:
        return ([], [-1], [])
    deviations = np.abs(kalman_flux - fluxes[::-1]) / kalman_error**.5
    #print(fluxes, f_var, kalman_flux, kalman_error**.5, deviations)
    # keep_idx_back = np.where(deviations < 500.)[0]
    keep_idx_back = np.where(deviations < 5.)[0]

    if len(keep_idx) >= len(keep_idx_back):
        new_lh = np.sum(psi_values[flux_idx[keep_idx]])/np.sqrt(np.sum(phi_values[flux_idx[keep_idx]]))
        return (val_on, flux_idx[keep_idx], new_lh)
    else:
        reverse_idx = len(flux_idx)-1 - keep_idx_back
        new_lh = np.sum(psi_values[flux_idx[reverse_idx]])/np.sqrt(np.sum(phi_values[flux_idx[reverse_idx]]))
        return (val_on, flux_idx[reverse_idx], new_lh)

def stamp_filter_parallel(stamps):

    s = stamps - np.min(stamps)
    s /= np.sum(s)
    s = np.array(s, dtype=np.dtype('float64')).reshape(21, 21)
    mom = measure.moments_central(s, cr=10, cc=10)
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
        keep_stamps = 1
    else:
        keep_stamps = 0

    return keep_stamps


class analysis_utils(object):

    def __init__(self):

        return


    def return_filename(self, visit_num):

        hits_file = 'v%i-fg.fits' % visit_num

        return hits_file

    def get_folder_visits(self, folder_visits):

        patch_visit_ids = np.array([int(visit_name[1:7]) for visit_name in folder_visits])

        return patch_visit_ids

    def load_images(self, im_filepath, time_file,psf_val=1.4, mjd_lims=None):
        """
        This function loads images and ingests them into a search object

        Input
        ---------

        im_filepath : string
            Image file path from which to load images

        time_file : string
            File name containing image times

        Output
        ---------

        search : kb.stack_search object

        ec_angle : ecliptic angle
        """

        visit_nums, visit_times = np.genfromtxt(time_file, unpack=True)
        image_time_dict = OrderedDict()
        for visit_num, visit_time in zip(visit_nums, visit_times):
            image_time_dict[str(int(visit_num))] = visit_time

        chunk_size = 100000

        start = time.time()

        patch_visits = sorted(os.listdir(im_filepath))
        patch_visit_ids = self.get_folder_visits(patch_visits)
        patch_visit_times = np.array([image_time_dict[str(visit_id)] for visit_id in patch_visit_ids])

        if mjd_lims is None:
            use_images = patch_visit_ids
        else:
            visit_only = np.where(((patch_visit_times > mjd_lims[0])
                                   & (patch_visit_times < mjd_lims[1])))[0]
            print(visit_only)
            use_images = patch_visit_ids[visit_only]

        image_mjd = np.array([image_time_dict[str(visit_id)] for visit_id in use_images])
        times = image_mjd - image_mjd[0]

        flags = ~0 # mask pixels with any flags
        flag_exceptions = [32,39] # unless it has one of these special combinations of flags
        master_flags = int('100111', 2) # mask any pixels which have any of
        # these flags in more than two images

        hdulist = fits.open('%s/%s' % (im_filepath, self.return_filename(use_images[0])))
        w = WCS(hdulist[1].header)
        ec_angle = self.calc_ecliptic_angle(w)
        ec_angle = np.pi + 1.25
        del(hdulist)

        images = [kb.layered_image('%s/%s' % (im_filepath, self.return_filename(f))) for f in np.sort(use_images)]

        print('Images Loaded')

        p = kb.psf(psf_val)
        stack = kb.image_stack(images)

        # Apply masks
        stack.apply_mask_flags(flags, flag_exceptions)
        stack.apply_master_mask(master_flags, 2)

        #stack.grow_mask()
        #stack.grow_mask()

        #stack.apply_mask_threshold(120.)

        stack.set_times(times)
        print("Times set")

        x_size = stack.get_width()
        y_size = stack.get_width()

        search = kb.stack_search(stack, p)

        return(search,ec_angle)

    def process_results(search):

        keep = {'stamps': [], 'new_lh': [], 'results': [], 'times': [],
                'lc': [], 'final_results': []}
        likelihood_limit = False
        res_num = 0
        chunk_size = 500000
        print('---------------------------------------')
        print("Processing Results")
        print('---------------------------------------')
        while likelihood_limit is False:
            pool = mp.Pool(processes=16)
            results = search.get_results(res_num,chunk_size)
            chunk_headers = ("Chunk Start", "Chunk Size", "Chunk Max Likelihood",
                             "Chunk Min. Likelihood")
            chunk_values = (res_num, len(keep['results']), results[0].lh, results[-1].lh)
            for header, val, in zip(chunk_headers, chunk_values):
                if type(val) == np.int:
                    print('%s = %i' % (header, val))
                else:
                    print('%s = %.2f' % (header, val))
            print('---------------------------------------')
            psi_curves = []
            phi_curves = []
            print(search.get_results(0,10))
            for line in results:
                psi_curve, phi_curve = search.lightcurve(line)
                psi_curves.append(np.array(psi_curve).flatten())
                phi_curve = np.array(phi_curve).flatten()
                phi_curve[phi_curve == 0.] = 99999999.
                phi_curves.append(phi_curve)
                if line.lh < likelihood_level:
                    likelihood_limit = True
                    break
            keep_idx_results = pool.starmap_async(return_indices,
                                                  zip(psi_curves, phi_curves,
                                                      [j for j in range(len(psi_curves))]))
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
                elif keep_idx_results[result_on][2] < likelihood_level:
                    continue
                else:
                    keep_idx = keep_idx_results[result_on][1]
                    new_likelihood = keep_idx_results[result_on][2]
                    keep['results'].append(results[result_on])
                    keep['new_lh'].append(new_likelihood)
                    stamps = search.sci_stamps(results[result_on], 10)
                    stamp_arr = np.array([np.array(stamps[s_idx]) for s_idx in keep_idx])
                    keep['stamps'].append(np.sum(stamp_arr, axis=0))
                    keep['lc'].append((psi_curves[result_on]/phi_curves[result_on])[keep_idx])
                    keep['times'].append(image_mjd[keep_idx])
            print(len(keep['results']))

            if len(keep['results']) > 500000:
                with open('%s/memory_error_tr_%i_patch_%s.txt' %
                          (res_filepath, tract, patch), 'w') as f:
                    f.write('In %i total results, %i were kept. Needs manual look.' %
                            (res_num + chunk_size, len(keep['results'])))
                memory_error = True
                likelihood_limit = True

            if res_num+chunk_size >= 4000000:
                likelihood_level = 20.
                with open('%s/overload_error_tr_%i_patch_%s.txt' %
                          (res_filepath, tract, patch), 'w') as f:
                    f.write('In %i total results, %i were kept. Likelihood level down to %f.' %
                            (res_num + chunk_size, len(keep['results']), line.lh))

            res_num += chunk_size

            return(keep)

    def cluster_results(self,keep):
        lh_sorted_idx = np.argsort(np.array(keep['new_lh']))[::-1]

        print(len(lh_sorted_idx))
        if len(lh_sorted_idx) > 0:
            print("Stamp filtering %i results" % len(lh_sorted_idx))
            pool = mp.Pool(processes=16)
            stamp_filt_pool = pool.map_async(stamp_filter_parallel,
                                             np.array(keep['stamps'])[lh_sorted_idx])
            pool.close()
            pool.join()
            stamp_filt_results = stamp_filt_pool.get()
            stamp_filt_idx = lh_sorted_idx[np.where(np.array(stamp_filt_results) == 1)]
            if len(stamp_filt_idx) > 0:
                print("Clustering %i results" % len(stamp_filt_idx))
                cluster_idx = self.cluster_results(np.array(keep['results'])[stamp_filt_idx],
                                                   x_size, y_size, [vel_min, vel_max],
                                                   [ang_min, ang_max])
                keep['final_results'] = stamp_filt_idx[cluster_idx]
            else:
                cluster_idx = []
                keep['final_results'] = []
            del(cluster_idx)
            del(stamp_filt_results)
            del(stamp_filt_idx)
            del(stamp_filt_pool)
        else:
            final_results = lh_sorted_idx

        print('Keeping %i results' % len(keep['final_results']))
        return(keep)

    def save_results(self, res_filepath, out_suffix, keep):
        """
        Save results from a given search method
        (either region search or grid search)

        Input
        --------

        res_filepath : string

        out_suffix : string
            Suffix to append to the output file name

        keep : dictionary
            Dictionary that contains the values to keep and print to filtering
        """

        np.savetxt('%s/results_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep['results'])[keep['final_results']], fmt='%s')
        with open('%s/lc_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['lc'])[keep['final_results']])
        with open('%s/times_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['times'])[keep['final_results']])
        np.savetxt('%s/filtered_likes_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep['new_lh'])[keep['final_results']], fmt='%.4f')
        np.savetxt('%s/ps_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep['stamps']).reshape(len(keep['stamps']), 441)[keep['final_results']], fmt='%.4f')

    def cluster_results(self, results, x_size, y_size,
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

    def calc_ecliptic_angle(self, test_wcs, angle_to_ecliptic=0.):

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
