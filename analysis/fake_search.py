import os
import shutil
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import astropy.coordinates as astroCoords
import astropy.units as u
import csv
import trajectoryFiltering as tf
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from skimage import measure
from analysis_utils import analysis_utils, \
    return_indices, stamp_filter_parallel
from collections import OrderedDict


class run_search(analysis_utils):

    def __init__(self, v_list, ang_list, num_obs):

        """
        Input
        --------

        v_list : list

            [min_velocity, max_velocity, velocity_steps]

        ang_list: list

            [radians below ecliptic,
             radians above ecliptic,
             steps]

        num_obs : integer

            Number of images a trajectory must be unmasked.
        """
        
        self.v_arr = np.array(v_list)
        self.ang_arr = np.array(ang_list)
        self.num_obs = num_obs

        return

    def run_search(self, im_filepath, res_filepath, out_suffix, time_file,
                   likelihood_level=10., mjd_lims=None, num_fakes=25,
                   rand_seed=42):

        visit_nums, visit_times = np.genfromtxt(time_file, unpack=True)
        image_time_dict = OrderedDict()
        for visit_num, visit_time in zip(visit_nums, visit_times):
            image_time_dict[str(int(visit_num))] = visit_time

        chunk_size = 100000

        start = time.time()
        
        patch_visits = sorted(os.listdir(im_filepath))
        patch_visit_ids = np.array([int(visit_name[1:7]) for visit_name in patch_visits])
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
            
        hdulist = fits.open('%s/v%i-fg.fits' % (im_filepath, use_images[0]))
        f0 = hdulist[0].header['FLUXMAG0']
        w = WCS(hdulist[1].header)
        ec_angle = self.calc_ecliptic_angle(w)
        del(hdulist)

        images = [kb.layered_image('%s/v%i-fg.fits' % (im_filepath, f)) for f in np.sort(use_images)]
        print('Images Loaded')

        p = kb.psf(1.4)

        # Add fakes steps
        print('Adding fake objects')
        x_fake_range = (5, 3650)
        y_fake_range = (5, 3650)
        angle_range = (ec_angle-(np.pi/15.), ec_angle+(np.pi/15.))
        velocity_range = (100, 500)
        mag_range = (20, 26)

        fake_results = []
        fake_output = []

        np.random.seed(rand_seed)
        for val in range(num_fakes):
            traj = kb.trajectory()
            traj.x = int(np.random.uniform(*x_fake_range))
            traj.y = int(np.random.uniform(*y_fake_range))
            ang = np.random.uniform(*angle_range)
            vel = np.random.uniform(*velocity_range)
            traj.x_v = vel*np.cos(ang)
            traj.y_v = vel*np.sin(ang)
            mag_val = np.random.uniform(*mag_range)
            traj.flux = f0*np.power(10, -0.4*mag_val)
            fake_results.append(traj)
            fake_output.append([traj.x, traj.y, traj.x_v, traj.y_v, traj.flux, mag_val])

        for fake_obj in fake_results:
            tf.add_trajectory(images, fake_obj, p, times)
        
        stack = kb.image_stack(images)
        del(images)
        stack.apply_mask_flags(flags, flag_exceptions)
        stack.apply_master_mask(master_flags, 2)
            
        stack.grow_mask()
        stack.grow_mask()

        stack.apply_mask_threshold(120.)

        stack.set_times(times)
        print("Times set")
        x_size = stack.get_width()
        y_size = stack.get_width()
        
        search = kb.stack_search(stack, p)
        del(stack)
        ang_min = ec_angle - self.ang_arr[0]
        ang_max = ec_angle + self.ang_arr[1]
        vel_min = self.v_arr[0]
        vel_max = self.v_arr[1]
        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("Ecliptic Angle", "Min. Search Angle", "Max Search Angle",
                         "Min Velocity", "Max Velocity")
        param_values = (ec_angle, ang_min, ang_max, vel_min, vel_max)
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        search.gpu(int(self.ang_arr[2]),int(self.v_arr[2]),ang_min,ang_max,
                   vel_min,vel_max,int(self.num_obs))

        keep_stamps = []
        keep_new_lh = []
        keep_results = []
        keep_times = []
        memory_error = False
        keep_lc = []
            
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
            chunk_values = (res_num, len(keep_results), results[0].lh, results[-1].lh)
            for header, val, in zip(chunk_headers, chunk_values):
                if type(val) == np.int:
                    print('%s = %i' % (header, val))
                else:
                    print('%s = %.2f' % (header, val))
            print('---------------------------------------')
            psi_curves = []
            phi_curves = []
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
                    keep_results.append(results[result_on])
                    keep_new_lh.append(new_likelihood)
                    stamps = search.sci_stamps(results[result_on], 10)
                    stamp_arr = np.array([np.array(stamps[s_idx]) for s_idx in keep_idx])
                    keep_stamps.append(np.sum(stamp_arr, axis=0))
                    keep_lc.append((psi_curves[result_on]/phi_curves[result_on])[keep_idx])
                    #keep_times.append(image_mjd[keep_idx])
                    keep_times.append(keep_idx)

            # if len(keep_results) > 800000:
            #     with open('%s/memory_error_tr_%s.txt' %
            #               (res_filepath, out_suffix), 'w') as f:
            #         f.write('In %i total results, %i were kept. Needs manual look.' %
            #                 (res_num + chunk_size, len(keep_results)))
            #     memory_error = True
            #     likelihood_limit = True
                    
            # if res_num+chunk_size >= 8000000:
            #     likelihood_level = 20.
            #     with open('%s/overload_error_tr_%s.txt' %
            #               (res_filepath, out_suffix), 'w') as f:
            #         f.write('In %i total results, %i were kept. Likelihood level down to %f.' %
            #                 (res_num + chunk_size, len(keep_results), line.lh))

            res_num += chunk_size

        del(search)

        lh_sorted_idx = np.argsort(np.array(keep_new_lh))[::-1]

        if len(lh_sorted_idx) > 0:
            print("Stamp filtering %i results" % len(lh_sorted_idx))
            pool = mp.Pool(processes=16)
            stamp_filt_pool = pool.map_async(stamp_filter_parallel,
                                             np.array(keep_stamps)[lh_sorted_idx])
            pool.close()
            pool.join()
            stamp_filt_results = stamp_filt_pool.get()
            stamp_filt_idx = lh_sorted_idx[np.where(np.array(stamp_filt_results) == 1)]
            if len(stamp_filt_idx) > 0:
                print("Clustering %i results" % len(stamp_filt_idx))
                cluster_idx = self.cluster_results(np.array(keep_results)[stamp_filt_idx],
                                                   x_size, y_size, [vel_min, vel_max],
                                                   [ang_min, ang_max])
                final_results = stamp_filt_idx[cluster_idx]
            else:
                cluster_idx = []
                final_results = []
            del(cluster_idx)
            del(stamp_filt_results)
            del(stamp_filt_idx)
            del(stamp_filt_pool)
        else:
            final_results = lh_sorted_idx            

        print('Keeping %i results' % len(final_results))
        
        np.savetxt('%s/results_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep_results)[final_results], fmt='%s')
        np.savetxt('%s/results_fakes_%s.txt' % (res_filepath, out_suffix),
                   np.array(fake_output), header='x,y,xv,yv,flux,mag')
        # np.savetxt('%s/lc_%s.txt' % (res_filepath, out_suffix),
        #            np.array(keep_lc)[final_results], fmt='%s')
        with open('%s/lc_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep_lc)[final_results])
        # np.savetxt('%s/times_%s.txt' % (res_filepath, out_suffix),
        #            np.array(keep_times)[final_results], fmt='%s')
        with open('%s/times_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep_times)[final_results])
        np.savetxt('%s/filtered_likes_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep_new_lh)[final_results], fmt='%.4f')
        np.savetxt('%s/ps_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep_stamps).reshape(len(keep_stamps), 441)[final_results], fmt='%.4f')

        end = time.time()

        del(keep_stamps)
        del(keep_times)
        del(keep_results)
        del(keep_new_lh)
        del(keep_lc)
                
        print("Time taken for patch: ", end-start)
