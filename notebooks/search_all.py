import keras
import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import subprocess
from astropy.io import fits
from sklearn.cluster import DBSCAN
from analyzeImage import analyzeImage
from keras.models import load_model

def write_param_file(real_image_path, psi_image_path, phi_image_path, results_name):
    paramsFile = open('../code/gpu/debug/parameters.config', 'w')
    paramsFile.write(
    """Debug ................ : 1
    PSF Sigma ............ : 1.0
    Mask Threshold ....... : 0.75
    Mask Penalty ......... : -0.05
    Slope Reject Threshold : 1000.0
    Angles to Search ..... : 120
    Minimum Angle ........ : 0.0
    Maximum Angle ........ : 6.283
    Velocities to Search . : 90
    Minimum Velocity ..... : 24.
    Maximum Velocity ..... : 600.
    Psi/Phi to file ...... : 1
    Source Images Path ... : ../../{source}/
    Psi Images Path ...... : ../../{psi}/
    Phi Images Path....... : ../../{phi}/
    Results Path ......... : ../../../data/results/{name}.txt
    """.format( source=real_image_path, psi=psi_image_path, phi=phi_image_path, name=results_name))
    paramsFile.close()

def trim_on_lc(clustered_results, im_array, image_times):

    f_results = clustered_results
    keep_results = []
    light_curves = []
    ai_2 = analyzeImage()

    for current in range(len(f_results)):
        traj_coords = ai_2.calc_traj_coords(f_results[current], image_times)
        light_curve = [im_array[x, traj_coords[x,1], traj_coords[x,0]] for x in range(len(im_array))]
        if np.max(light_curve) < 10*np.median(light_curve):
            keep_results.append(current)
            light_curves.append(light_curve)
            
    return f_results[keep_results], light_curves

def create_stamps(im_array, kept_results, image_times):

    f_results = kept_results#filtered_results
    postage_stamps = []
    ai_2 = analyzeImage()
    for imNum in range((len(f_results))):
        current = imNum#best_targets[imNum]
        ps = ai_2.createPostageStamp(im_array,
                                     list(f_results[['t0_x', 't0_y']][current]),
                                     np.array(list(f_results[['v_x', 'v_y']][current])),
                                     image_times, [25., 25.])
        postage_stamps.append(ps[0])

    return postage_stamps

def process_results():
    #Load results
    raw_results = np.genfromtxt('../data/results/HITS1.txt', names=True)
    print len(raw_results)
    #Load times
    image_mjd = []

    for filename in sorted(os.listdir(real_image_path)):
        hdulist = fits.open(os.path.join(real_image_path, filename))
        image_mjd.append(hdulist[0].header['MJD'])
        
    image_mjd = np.array(image_mjd)
    image_times = image_mjd - image_mjd[0]
        
    #Load images
    hdulist = fits.open(os.path.join(real_image_path, os.listdir(real_image_path)[0]))
    num_images = len(os.listdir(real_image_path))
    image_shape = np.shape(hdulist[1].data)
    im_array = np.zeros((num_images, image_shape[0], image_shape[1]))
    
    for idx, filename in list(enumerate(sorted(os.listdir(real_image_path)))):
        
        # print( str('Loaded ' + filename))
        image_file = os.path.join(real_image_path, filename)
        hdulist = fits.open(image_file)
        im_array[idx] = hdulist[1].data#*mask
            
    ai = analyzeImage()
        
    model = load_model('../data/kbmod_model.h5')
        
    results = raw_results[np.where(raw_results['likelihood'] >= 5.0)]
    print len(results)

    filtered_results = ai.filter_results(im_array, results, image_times, model, chunk_size=5000)
            
    results_to_cluster = filtered_results
    if len(filtered_results) > 0:
        chunk_size = 25000.
        if len(filtered_results) > chunk_size:
            chunk_results = []
            total_chunks = np.ceil(len(filtered_results)/chunk_size)
            print 'Dividing into %f chunks' % total_chunks
            chunk_on = 1
            for chunk_start in range(0, len(filtered_results), np.int(chunk_size)):
                print 'On chunk %i' % chunk_on
                chunk_on += 1
                arg = dict(eps=0.03, min_samples=1, n_jobs=-1)
                chunk_cluster_results = ai.clusterResults(results_to_cluster[chunk_start:chunk_start+np.int(chunk_size)],
                                                      dbscan_args=arg)#, im_array, image_times)
                chunk_results.append(np.array(chunk_cluster_results[1], dtype=np.int)+chunk_start)
            results_list = []
            for chunk_result in chunk_results:
                for ind_result in chunk_result:
                    results_list.append(ind_result)
            clustered_results = results_to_cluster[results_list]
            #clustered_results =  results_to_cluster[np.array(clustered_results[1], dtype=np.int)]
        else:
            arg = dict(eps=0.03, min_samples=1, n_jobs=-1)
            clustered_results = ai.clusterResults(results_to_cluster, dbscan_args=arg)#, im_array, image_times)
            clustered_results =  results_to_cluster[np.array(clustered_results[1], dtype=np.int)]
        print len(clustered_results)
            
        kept_results, light_curves = trim_on_lc(clustered_results, im_array, image_times)
        stamps = create_stamps(im_array, kept_results, image_times)
    else:
        kept_results = []
        light_curves = []
        stamps = []

    return kept_results, light_curves, stamps, image_times


if __name__ == "__main__":

    chip = sys.argv[1]
    sys.path.append('/home/kbmod-usr/cuda-workspace/kbmod/code/gpu/debug/')
    
    for chip_num in [chip]:#, '03', '04', '05']:
#        chip_results = []
#        chip_lc = []
#        chip_stamps = []
#        chip_times = []
#        field_id = []
#        df = pd.DataFrame(columns=['t0_x', 't0_y', 'theta_par', 'theta_perp', 'v_x',
#                               'v_y', 'likelihood', 'est_flux', 'field_num'])
        for field_num in xrange(1,57):
            df = pd.DataFrame(columns=['t0_x', 't0_y', 'theta_par', 'theta_perp', 'v_x',
                                       'v_y', 'likelihood', 'est_flux', 'field_num', 'times'])
            if field_num < 10:
                real_image_path = str("../../HITS/trimmed_chip_" + str(chip_num) +
                                      "/Blind15A_0" + str(field_num) + "/search_nights")
            else:
                real_image_path = str("../../HITS/trimmed_chip_" + str(chip_num) +
                                      "/Blind15A_" + str(field_num) + "/search_nights")
            #Test to make sure images are present for that night
            try:
                os.listdir(real_image_path)[0]
            except:
                continue

            results_name = "HITS1"
            gpu_code_path = "../code/gpu/"
            psi_image_path = gpu_code_path+"output-images/psi"
            phi_image_path = gpu_code_path+"output-images/phi"

            write_param_file(real_image_path, psi_image_path, phi_image_path, results_name)

            for file_name in os.listdir('../code/gpu/output-images/psi'):
                psi_path = os.path.join('../code/gpu/output-images/psi', file_name)
                try:
                    if os.path.isfile(psi_path):
                        os.unlink(psi_path)
                except:
                    continue
            for file_name in os.listdir('../code/gpu/output-images/phi'):
                phi_path = os.path.join('../code/gpu/output-images/phi', file_name)
                try:
                    if os.path.isfile(phi_path):
                        os.unlink(phi_path)
                except:
                    continue

            os.chdir('/home/kbmod-usr/cuda-workspace/kbmod/code/gpu/debug/')
            subprocess.call("./CudaTracker", shell=True)
            os.chdir(os.environ['PWD'])

            kept_results, light_curves, stamps, image_times = process_results()
            chip_lc = []
            chip_stamps = []
            if len(kept_results) > 0:
                field_df = pd.DataFrame.from_records(kept_results)
                field_df['field_num'] = field_num
                field_df['times'] = [image_times]*len(kept_results)
                df = df.append(field_df)
                for lc, stamp in zip(light_curves, stamps):
                    chip_lc.append(lc)
                    chip_stamps.append(np.reshape(stamp, (625,1)))
                    #chip_times.append(image_times)
                    #field_id.append(field_num)
            df.to_csv(str('results/data/' + str(chip_num) + '_' + str(field_num) + '_results.csv'), index=False)
            np.savetxt(str('results/stamps/' + str(chip_num) + '_' + str(field_num) + '_stamps.dat'), chip_stamps)
            np.savetxt(str('results/light_curves/' + str(chip_num) + '_' + str(field_num) + '_lightcurves.dat'), chip_lc)
            
            #except:
            #    continue

        #df.to_csv(str(str(chip_num) + '_results.csv'), index=False)
        #fig = plt.figure(figsize=(8, 3*len(field_id)))
        #for lc, stamp, plot_num, image_time_set in zip(chip_lc, chip_stamps, np.arange(len(field_id)), chip_times):
        #    fig.add_subplot(len(field_id),2,2*plot_num + 1)
        #    plt.imshow(stamp, origin='lower', interpolation='None')
        #    plt.title(str(field_id[plot_num]))
        #    fig.add_subplot(len(field_id),2,2*plot_num + 2)
        #    plt.plot(image_time_set, lc)
        #    plt.xlabel('Time (days)')
        #    plt.ylabel('Flux')
        #    plt.tight_layout()
        #plt.savefig(str(str(chip_num) + '_stamps.pdf'))
