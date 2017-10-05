import os
import pickle
import sys
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from skimage import measure
from filter_utils import *

if __name__ == "__main__":

    image_folder = sys.argv[1]
    results_file = sys.argv[2]

    # Following sets up ability to create psi/phi and is from
    # HITS_Main_Belt_Example.ipynb
    flags = ~0
    flag_exceptions = [32, 39]
    master_flags = int('100111', 2)

    image_file_list = [str(image_folder + '/' + filename) for filename in os.listdir(image_folder)]
    image_file_list.sort()
    images = [kb.layered_image(f) for f in image_file_list]
    p = kb.psf(1.4)
    stack = kb.image_stack(images)
    stack.apply_mask_flags(flags, flag_exceptions)
    stack.apply_master_mask(master_flags, 2)

    image_array = stack.get_images()
    search = kb.stack_search(stack, p)

    search.gpu(1, 2, -0.0442959674533, 0.741102195944, 1920.0, 4032.0, 3)

    psi = search.get_psi()
    phi = search.get_phi()

    image_times = np.array(stack.get_times())

    file_len, header_len = file_and_header_length(results_file)

    chunk_size = 250000
    total_chunks = 20
    print(total_chunks)

    results_keep = []
    results_ps = []
    results_curves = []
    results_likes = []

    for i in range(0, int(total_chunks)):
        print(i)
        results_arr = load_chunk(results_file, chunk_size*i + header_len, chunk_size)
        if results_arr['likelihood'][0] < 5.0:
            print('Likelihood now below 5.0. Breaking Loop.')
            break
        psi_lc, phi_lc = get_likelihood_lcs(results_arr, psi, phi, image_times)
        print('Kalman Filtering')
        pool = mp.Pool(processes=16)
        results = [pool.apply_async(create_filter_curve,
                                    args=(psi_lc[j*100:(j*100)+100],
                                          phi_lc[j*100:(j*100)+100],
                                          j)) for j in range(int(len(psi_lc)/100))]
        pool.close()
        pool.join()
        results = [p.get() for p in results]
        results.sort()
        new_nu = [x for y in results for x in y[1]]
        print('Reordering')
        reorder = np.argsort(new_nu)[::-1]
        reorder_keep = reorder[np.where(np.array(new_nu)[reorder] > 5.0)[0]]
        print('Clustering %i objects' % len(reorder_keep))
        if len(reorder_keep) > 0:
            db_cluster, top_vals = clusterResults(results_arr[reorder_keep])
            keep_details = results_arr[reorder_keep[top_vals]]
            keep_new_nu = np.array(new_nu)[reorder_keep[top_vals]]
            keep_ps = []
            keep_lc = []
            for i in range(len(top_vals)):
                obj_num = reorder_keep[top_vals[i]]
                kalman_curve, kalman_idx = return_filter_curve(psi_lc[obj_num], phi_lc[obj_num])

                ps = createPostageStamp(np.array(image_array)[kalman_idx], results_arr[['t0_x', 't0_y']][obj_num],
                                        results_arr[['v_x', 'v_y']][obj_num], image_times[kalman_idx],
                                        [21., 21.])

                ps_1 = np.array(ps[1])
                ps_1[ps_1 < -9000] = 0.
                keep_ps.append(np.sum(np.array(ps_1), axis=0))
                keep_lc.append([kalman_idx, kalman_curve])

            results_keep.append(keep_details)
            results_ps.append(keep_ps)
            results_curves.append(keep_lc)
            results_likes.append(keep_new_nu)
            print('Returning %i objects' % len(keep_details))

        # Get the results out of lists of lists and into single lists
    final_results = [x for y in results_keep for x in y]
    final_ps = [x for y in results_ps for x in y]
    final_curves = [x for y in results_curves for x in y]
    final_likelihoods = [x for y in results_likes for x in y]
    
    new_f = np.array(final_results, dtype=[('t0_x', np.float), ('t0_y', np.float), ('v_x', np.float),
                                           ('v_y', np.float), ('likelihood', np.float), ('est_flux', np.float)])
    
    # Cluster again on full returned results
    db_cluster, top_vals = clusterResults(new_f)
    
    final_results = new_f[top_vals]
    final_ps = np.array(final_ps)[top_vals]
    final_curves = np.array(final_curves)[top_vals]
    final_likelihoods = np.array(final_likelihoods)[top_vals]
    
    # Filtering step
    mom_array = []
    for s in final_ps:
        s[s < -9000] = 0.
        # Normalize flux in each stamp
        s -= np.min(s)
        s /= np.sum(s)
        s = np.array(s, dtype=np.float)
        mom = measure.moments_central(s, cr=10, cc=10)
        mom_array.append([mom[2,0], mom[0,2], mom[1,1], mom[1,0], mom[0,1]])
    mom_array = np.array(mom_array)

    # These are the filtering parameters, but could be fine tuned still
    mom_array_test = np.where((mom_array[:,0] < 32.) & (mom_array[:,1] < 32.)  &
                              (np.abs(mom_array[:,3]) < .3) & (np.abs(mom_array[:,4]) < .3))[0]
    
    final_results = final_results[mom_array_test]
    final_ps = np.array(final_ps)[mom_array_test]
    final_curves = np.array(final_curves)[mom_array_test]
    final_likelihoods = np.array(final_likelihoods)[mom_array_test]
    
    # Save results
    np.savetxt('final_results.csv', final_results)
    np.savetxt('final_ps.dat', final_ps.reshape(len(final_results), 21*21))
    with open('final_curves.pkl', 'wb') as fi:
        pickle.dump(final_curves, fi)
    np.savetxt('final_likelihoods.dat', final_likelihoods)
