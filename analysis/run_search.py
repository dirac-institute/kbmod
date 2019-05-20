import os
import sys
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
from analysis_utils import Interface, PostProcess

class region_search:
    """
    CLASS CURRENTLY DOES NOT WORK
    """
    def __init__(self,v_guess,radius,num_obs):
        """
        INPUT-
            v_guess : float array
                Initial object velocity guess. Given as an array or tuple.
                Algorithm will search velocities within 'radius' of 'v_guess'
            radius : float
                radius in velocity space to search, centered around 'v_guess'
            num_obs : int
                The minimum number of observations required to keep the object
        """
        self.v_guess = v_guess
        self.radius = radius
        self.num_obs = num_obs
        return

    def run_search(self, im_filepath, res_filepath, out_suffix, time_file,
                   likelihood_level=10., mjd_lims=None):
        # Initialize some values
        start = time.time()

        memory_error = False
        # Load images to search
        search,image_params = self.load_images(
            im_filepath, time_file, mjd_lims=mjd_lims)

        # Run the region search
        # Save values in image_params for use in filter_results

        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("X Velocity Guess","Y Velocity Guess",
                         "Radius in velocity space")
        param_values = (*self.v_guess,self.radius)
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        results = search.region_search(
            *self.v_guess, self.radius, likelihood_level, int(self.num_obs))
        duration = image_params['times'][-1]-image_params['times'][0]
        # Convert the results to the grid formatting
        grid_results = kb.region_to_grid(results,duration)
        # Process the search results
        keep = self.process_region_results(
            search, image_params, res_filepath, likelihood_level, grid_results)
        del(search)

        # Cluster the results
        #keep = self.filter_results(keep,image_params)

        # Save the results
        self.save_results(res_filepath, out_suffix, keep)

        end = time.time()

        del(keep)
        return

    def process_region_results(
        self,search,image_params,res_filepath,likelihood_level,results):
        """
        Processes results that are output by the gpu search.
        """

        keep = {'stamps': [], 'new_lh': [], 'results': [], 'times': [],
                'lc': [], 'final_results': []}

        print('---------------------------------------')
        print("Processing Results")
        print('---------------------------------------')
        print('Starting pooling...')
        pool = mp.Pool(processes=16)
        print('Getting results...')

        psi_curves = []
        phi_curves = []
        # print(results)
        for line in results:
            psi_curve, phi_curve = search.lightcurve(line)
            psi_curves.append(np.array(psi_curve).flatten())
            phi_curve = np.array(phi_curve).flatten()
            phi_curve[phi_curve == 0.] = 99999999.
            phi_curves.append(phi_curve)

        keep_idx_results = pool.starmap_async(
            return_indices,
            zip(psi_curves, phi_curves, [j for j in range(len(psi_curves))]))
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()
        if (len(keep_idx_results) < 1):
            keep_idx_results = [(0,[-1],0.)]

        if (len(keep_idx_results[0]) < 3):
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
                stamp_arr = np.array(
                    [np.array(stamps[s_idx]) for s_idx in keep_idx])
                keep['stamps'].append(np.sum(stamp_arr, axis=0))
                keep['lc'].append(
                    (psi_curves[result_on]/phi_curves[result_on])[keep_idx])
                keep['times'].append(image_params['mjd'][keep_idx])
        print(len(keep['results']))
        # Needed for compatibility with grid_search save functions
        keep['final_results'] = range(len(keep['results']))

        return(keep)

class run_search:
    """
    This class runs the grid search for kbmod.
    """
    def __init__(self, v_list, ang_list, num_obs):

        """
        INPUT-
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

    def do_gpu_search(self, search, image_params):

        # Run the grid search
        # Set min and max values for angle and velocity
        search_start = time.time()
        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("Ecliptic Angle", "Min. Search Angle",
                         "Max Search Angle", "Min Velocity", "Max Velocity")
        param_values = (image_params['ec_angle'], *image_params['ang_lims'],
                        *image_params['vel_lims'])
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        search.gpu(
            int(self.ang_arr[2]), int(self.v_arr[2]),
            *image_params['ang_lims'], *image_params['vel_lims'],
            int(self.num_obs))
        print(
            'Search finished in {0:.3f}s'.format(time.time()-search_start),
            flush=True)
        return(search, image_params)

    def run_search(
        self, im_filepath, res_filepath, out_suffix, time_file, lh_level=10.,
        psf_val=1.4, mjd_lims=None, average_angle=None):
        """
        This function serves as the highest-level python interface for starting
        a KBMOD search.
        INPUT-
            im_filepath : string
                Path to the folder containing the images to be ingested into
                KBMOD and searched over.
            res_filepath : string
                Path to the folder that will contain the results from the
                search.
            out_suffix : string
                Suffix to append to the output files. Used to differentiate
                between different searches over the same stack of images.
            time_file : string
                Path to the file containing the image times.
            lh_level : float
                Minimum acceptable likelihood level for a trajectory.
                Trajectories with likelihoods below this value will be
                discarded.
            psf_val : float
                Determines the size of the psf generated by the kbmod stack.
            mjd_lims : numpy array
                Limits the search to images taken within the limits input by
                mjd_lims.
            average_angle : float
                Overrides the ecliptic angle calculation and instead centers
                the average search around average_angle.
        """

        start = time.time()
        kb_interface = Interface()
        kb_post_process = PostProcess()

        # Load images to search
        stack,image_params = kb_interface.load_images(
            im_filepath, time_file, mjd_lims=mjd_lims)
        # Save values in image_params for later use
        stack = kb_post_process.apply_mask(stack, mask_num_images=25)
        psf = kb.psf(psf_val)
        search = kb.stack_search(stack, psf)

        if average_angle == None:
            average_angle = image_params['ec_angle']
        ang_min = average_angle - self.ang_arr[0]
        ang_max = average_angle + self.ang_arr[1]
        vel_min = self.v_arr[0]
        vel_max = self.v_arr[1]
        image_params['ang_lims'] = [ang_min, ang_max]
        image_params['vel_lims'] = [vel_min, vel_max]
        search, image_params = self.do_gpu_search(
            search, image_params)
        # Load the KBMOD results into Python and apply a filter based on
        # 'filter_type'
        keep = kb_post_process.load_results(
            search, image_params, lh_level, filter_type='clipped_sigmaG',
            max_lh=100)
        keep = kb_post_process.get_coadd_stamps(keep, search)
        keep = kb_post_process.apply_stamp_filter(keep)
        keep = kb_post_process.apply_clustering(keep, image_params)
        keep = kb_post_process.get_all_stamps(keep, search)
        del(search)
        # Save the results
        kb_interface.save_results(res_filepath, out_suffix, keep)
        end = time.time()
        del(keep)

        print("Time taken for patch: ", end-start)
