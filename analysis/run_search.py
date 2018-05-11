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

class run_region_search(analysis_utils):

    def __init(self,v_guess,radius):

        """
        Input
        --------

        v_guess : float

            Initial object velocity guess. Algorithm will search velocities
            within 'radius' of 'v_guess'

        radius : float

            radius in velocity space to search, centered around 'v_guess'
        """

        return

    def run_search(self, im_filepath, res_filepath, out_suffix, time_file,
                   likelihood_level=10., mjd_lims=None):
        # Load images to search
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

        memory_error = False

        # Load images to search
        search,ec_angle = self.load_images(im_filepath,time_file,mjd_lims=mjd_lims)

        # Run the grid search
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

        # Process the search results
        keep = self.process_results(search,likelihood_level)
        del(search)

        # Cluster the results
        keep = self.filter_results(keep)

        # Save the results
        self.save_results(res_filepath, out_suffix, keep)

        end = time.time()

        del(keep)

        print("Time taken for patch: ", end-start)
