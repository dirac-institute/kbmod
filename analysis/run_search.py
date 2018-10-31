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
from analysis_utils import analysis_utils, \
    return_indices, stamp_filter_parallel

class region_search(analysis_utils):

    def __init__(self,v_guess,radius,num_obs):

        """
        Input
        --------

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
        search,image_params = self.load_images(im_filepath,time_file,mjd_lims=mjd_lims)

        # Run the region search
        # Save values in image_params for use in filter_results

        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("X Velocity Guess","Y Velocity Guess","Radius in velocity space")
        param_values = (*self.v_guess,self.radius)
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        results = search.region_search(*self.v_guess, self.radius, likelihood_level, int(self.num_obs))
        duration = image_params['times'][-1]-image_params['times'][0]
        grid_results = kb.region_to_grid(results,duration) # Convert the results to the grid formatting
        # Process the search results
        keep = self.process_results(search,image_params,res_filepath,likelihood_level,results=grid_results)
        del(search)

        # Cluster the results
        #keep = self.filter_results(keep,image_params)

        # Save the results
        self.save_results(res_filepath, out_suffix, keep)

        end = time.time()

        del(keep)
        return

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
                   likelihood_level=10., mjd_lims=None):

        # Initialize some values
        start = time.time()

        memory_error = False
        # Load images to search
        search,image_params = self.load_images(im_filepath,time_file,mjd_lims=mjd_lims)

        # Run the grid search

        # Set min and max values for angle and velocity
        ang_min = image_params['ec_angle'] - self.ang_arr[0]
        ang_max = image_params['ec_angle'] + self.ang_arr[1]
        vel_min = self.v_arr[0]
        vel_max = self.v_arr[1]
        # Save values in image_params for use in filter_results
        image_params['ang_lims'] = [ang_min, ang_max]
        image_params['vel_lims'] = [vel_min, vel_max]

        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("Ecliptic Angle", "Min. Search Angle", "Max Search Angle",
                         "Min Velocity", "Max Velocity")
        param_values = (image_params['ec_angle'], *image_params['ang_lims'], *image_params['vel_lims'])
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        search.gpu(int(self.ang_arr[2]),int(self.v_arr[2]),*image_params['ang_lims'],
                   *image_params['vel_lims'],int(self.num_obs))

        # Process the search results
        keep = self.process_results(search,image_params,res_filepath,likelihood_level)
        del(search)

        # Cluster the results
        keep = self.filter_results(keep,image_params)

        # Save the results
        self.save_results(res_filepath, out_suffix, keep)

        end = time.time()

        del(keep)

        print("Time taken for patch: ", end-start)
