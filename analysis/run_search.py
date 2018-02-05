import os
import shutil
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import astropy.coordinates as astroCoords
import astropy.units as u
import getpass
from pexpect import pxssh
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from skimage import measure
from analysis_utils import analysis_utils


class run_search(analysis_utils):

    def __init__(self, v_list, ang_list, num_obs):

        self.v_arr = v_list
        self.ang_arr = ang_list
        self.num_obs = num_obs

        return

    def run_search(self, im_filepath, res_filepath, time_file):

        image_times = np.genfromtxt(time_file)
        image_time_dict = {str(int(times_row[0])):times_row[1] for times_row in image_times}

        chunk_size = 100000

        patch_dir = os.path.join(tract_path, patch)
        os.mkdir(patch_path)

        start = time.time()

        patch_visits = sorted(os.listdir(patch_dir))
        patch_visit_ids = np.array([int(visit_name[1:7]) for visit_name in patch_visits])

        # TODO: Change this to times
        visit_first_3 = np.where(((patch_visit_ids < 412000) & (patch_visit_ids > 410800)))[0]

            use_images = patch_visit_ids[visit_first_3]

            flags = ~0 # mask pixels with any flags
            flag_exceptions = [32,39] # unless it has one of these special combinations of flags
            master_flags = int('100111', 2) # mask any pixels which have any of 
            # these flags in more than two images
            
            hdulist = fits.open('%s/v%i-fg.fits' % (patch_dir, use_images[0]))
            w = WCS(hdulist[1].header)
            ec_angle = calc_ecliptic_angle(w)
            del(hdulist)

            likelihood_level = 10.

            images = [kb.layered_image('%s/v%i-fg.fits' % (patch_dir, f)) for f in np.sort(use_images)]
            print('Images Loaded')

            p = kb.psf(1.4)
            stack = kb.image_stack(images)
            del(images)
            stack.apply_mask_flags(flags, flag_exceptions)
            stack.apply_master_mask(master_flags, 2)
            
            stack.grow_mask()
            stack.grow_mask()

            stack.apply_mask_threshold(120.)
            
            image_mjd = np.array([image_time_dict[str(visit_id)] for visit_id in use_images])
            
            times = image_mjd - image_mjd[0]
            stack.set_times(times)
            print("Times set")
            x_size = stack.get_width()
            y_size = stack.get_width()
            
            #x_vel = 205.#(object_df['x_pixel'].values[-1] - object_df['x_pixel'].values[0])/times[-1]
            #y_vel = 70.#(object_df['y_pixel'].values[-1] - object_df['y_pixel'].values[0])/times[-1]

            search = kb.stack_search(stack, p)
            del(stack)
            ang_min = ec_angle-(np.pi/15.)
            ang_max = ec_angle+(np.pi/15.) 
            vel_min = 92.
            vel_max = 526.
            print("Starting Search")
            print(ec_angle, ang_min, ang_max)
            search.gpu(12,25,ang_min,ang_max,vel_min,vel_max,6)
            # search.gpu(128,250,ang_min,ang_max,vel_min,vel_max,6)
                
            # num_results = 10000000
            # num_chunks = np.ceil(num_results/chunk_size)
            keep_stamps = []
            keep_new_lh = []
            keep_results = []
            keep_times = []
            memory_error = False
            keep_lc = []
            
            likelihood_limit = False
            res_num = 0
            chunk_size = 500000
            while likelihood_limit is False:
                pool = mp.Pool(processes=16)
                results = search.get_results(res_num,chunk_size)
                print(res_num, len(keep_results), results[0].lh, results[-1].lh)
                # if results[-1].lh < likelihood_level:
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
                keep_idx_results = pool.starmap_async(return_indices, zip(psi_curves, phi_curves, [j for j in range(len(psi_curves))]))
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
                        keep_times.append(image_mjd[keep_idx])  

                if len(keep_results) > 500000:
                    with open('%s/memory_error_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), 'w') as f:
                        f.write('In %i total results, %i were kept. Needs manual look.' % (res_num + chunk_size, len(keep_results)))
                    memory_error = True
                    likelihood_limit = True
                    
                if res_num+chunk_size >= 4000000:
                    likelihood_level = 20.
                    with open('%s/overload_error_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), 'w') as f:
                        f.write('In %i total results, %i were kept. Likelihood level down to %f.' % (res_num + chunk_size, len(keep_results), line.lh))

                res_num += chunk_size

            del(search)

            lh_sorted_idx = np.argsort(np.array(keep_new_lh))[::-1]


            if len(lh_sorted_idx) > 0:
                print("Stamp filtering %i results" % len(lh_sorted_idx))
                pool = mp.Pool(processes=16)
                stamp_filt_pool = pool.map_async(stamp_filter_parallel, np.array(keep_stamps)[lh_sorted_idx])
                pool.close()
                pool.join()
                stamp_filt_results = stamp_filt_pool.get()
                stamp_filt_idx = lh_sorted_idx[np.where(np.array(stamp_filt_results) == 1)]
                if len(stamp_filt_idx) > 0:
                    print("Clustering %i results" % len(stamp_filt_idx))
                    cluster_idx = cluster_results(np.array(keep_results)[stamp_filt_idx],
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

            np.savetxt('%s/results_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), np.array(keep_results)[final_results], fmt='%s')
            np.savetxt('%s/lc_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), np.array(keep_lc)[final_results], fmt='%s')
            np.savetxt('%s/times_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), np.array(keep_times)[final_results], fmt='%s')
            np.savetxt('%s/filtered_likes_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), np.array(keep_new_lh)[final_results], fmt='%.4f')
            np.savetxt('%s/ps_tr_%i_patch_%s.txt' % (res_filepath, tract, patch), np.array(keep_stamps).reshape(len(keep_stamps), 441)[final_results], fmt='%.4f')

            end = time.time()

            del(keep_stamps)
            del(keep_times)
            del(keep_results)
            del(keep_new_lh)
            del(keep_lc)
                
            print("Time taken for patch: ", end-start)
        shutil.rmtree(tract_path)
