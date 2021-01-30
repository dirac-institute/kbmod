from __future__ import print_function

import ephem
import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import copysign
from scipy.stats import multivariate_normal
from pyOrbfit.Orbit import Orbit
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord


class KbmodInfo(object):
    """
    Hold and process the information describing the input data
    and results from a KBMOD search.
    """

    def __init__(self, results_filename, image_filename,
                 visit_list, visit_mjd, results_visits, observatory):

        """
        Read in the output from a KBMOD search and store as a pandas
        DataFrame.

        Take in a list of visits and times for those visits.

        Parameters
        ----------
        results_filename: str
            The filename of the kbmod search results.

        image_filename: str
            The filename of the first image used in the kbmod search.

        visit_list: numpy array
            An array with all possible visit numbers in the search fields.

        visit_mjd: numpy array
            An array of the corresponding times in MJD for the visits listed
            in `visit_list`.

        results_visits: list
            A list of the visits actually searched by kbmod for the given
            results file.

        observatory: str
            The three character observatory code for the data searched.
        """
        
        self.visit_df = pd.DataFrame(visit_list,
                                     columns=['visit_num'])
        self.visit_df['visit_mjd'] = visit_mjd

        results_array = np.genfromtxt(results_filename)

        # Only keep values and not property names from results file
        if len(np.shape(results_array)) == 1:
            results_proper = [results_array[1::2]]
        elif len(np.shape(results_array)) > 1:
            results_proper = results_array[:, 1::2]

        self.results_df = pd.DataFrame(results_proper,
                                       columns=['lh', 'flux', 'x0', 'y0',
                                                'x_v', 'y_v', 'obs_count'])

        image_fits = fits.open(image_filename)
        self.wcs = WCS(image_fits[1].header)

        self.results_visits = results_visits

        self.results_mjd = self.visit_df[self.visit_df['visit_num'].isin(self.results_visits)]['visit_mjd'].values
        self.mjd_0 = self.results_mjd[0]

        self.obs = observatory
        
    @staticmethod
    def mpc_reader(filename):

        """
        Read in a file with observations in MPC format and return the coordinates.

        Parameters
        ----------
        filename: str
            The name of the file with the MPC-formatted observations.

        Returns
        -------
        c: astropy SkyCoord object
            A SkyCoord object with the ra, dec of the observations.
        """
        iso_times = []
        time_frac = []
        ra = []
        dec = []

        with open(filename, 'r') as f:
            for line in f:
                year = str(line[15:19])
                month = str(line[20:22])
                day = str(line[23:25])
                iso_times.append(str('%s-%s-%s' % (year,month,day)))
                time_frac.append(str(line[25:31]))
                ra.append(str(line[32:44]))
                dec.append(str(line[44:56]))

        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

        return c
        
    def get_searched_radec(self, obj_idx):

        """
        This will take an image and use its WCS to calculate the
        ra, dec locations of the object in the searched data.

        Parameters
        ----------
        obj_idx: int
            The index of the object in the KBMOD results for which
            we want to calculate orbital elements/predictions.
        """

        self.result = self.results_df.iloc[obj_idx]

        zero_times = self.results_mjd - self.mjd_0

        pix_coords_x = self.result['x0'] + \
                       self.result['x_v']*zero_times
        pix_coords_y = self.result['y0'] + \
                       self.result['y_v']*zero_times

        ra, dec = self.wcs.all_pix2world(pix_coords_x, pix_coords_y, 1)

        self.coords = SkyCoord(ra*u.deg, dec*u.deg)
    
    def format_results_mpc(self):

        """
        This method will take in a row from the results file and output the
        astrometry of the object in the searched observations into a file with
        MPC formatting.
            
        Returns
        -------
        mpc_lines: list of strings
            List where each entry is an observation as an MPC-formatted string
        """

        field_times = Time(self.results_mjd, format='mjd')

        mpc_lines = []
        for t, c in zip(field_times, self.coords):
            mjd_frac = t.mjd % 1.0
            ra_hms = c.ra.hms
            dec_dms = c.dec.dms
            if dec_dms.d != 0:
                name = ("     c111112  c%4i %02i %08.5f %02i %02i %06.3f%+03i %02i %05.2f                     %s" %
                        (t.datetime.year, t.datetime.month, t.datetime.day+mjd_frac,
                         ra_hms.h, ra_hms.m, ra_hms.s,
                         dec_dms.d, np.abs(dec_dms.m), np.abs(dec_dms.s), self.obs))
            else:
                if copysign(1, dec_dms.d) == -1.0:
                    dec_dms_d = '-00'
                else:
                    dec_dms_d = '+00'
                name = ("     c111112  c%4i %02i %08.5f %02i %02i %06.3f%s %02i %05.2f                     %s" %
                        (t.datetime.year, t.datetime.month, t.datetime.day+mjd_frac,
                         ra_hms.h, ra_hms.m, ra_hms.s,
                         dec_dms_d, np.abs(dec_dms.m), np.abs(dec_dms.s), self.obs))

            mpc_lines.append(name)

        return(mpc_lines)
    
    def save_results_mpc(self, file_out):
        """
        Save the MPC-formatted observations to file.
        
        Parameters
        ----------
        file_out: str
            The output filename with the MPC-formatted observations
            of the KBMOD search result. If None, then it will save
            the output as 'kbmod_mpc.dat' and will be the default
            file in other methods below where file_in=None.
        """
        
        mpc_lines = self.format_results_mpc()
        with open(file_out, 'w') as f:
            for obs in mpc_lines:
                f.write(obs + '\n')
                
    def get_searched_radec(self, obj_idx):

        """
        This will take an image and use its WCS to calculate the
        ra, dec locations of the object in the searched data.

        Parameters
        ----------
        obj_idx: int
            The index of the object in the KBMOD results for which
            we want to calculate orbital elements/predictions.
        """

        self.result = self.results_df.iloc[obj_idx]

        zero_times = self.results_mjd - self.mjd_0

        pix_coords_x = self.result['x0'] + \
                       self.result['x_v']*zero_times
        pix_coords_y = self.result['y0'] + \
                       self.result['y_v']*zero_times

        ra, dec = self.wcs.all_pix2world(pix_coords_x, pix_coords_y, 1)

        self.coords = SkyCoord(ra*u.deg, dec*u.deg)
                
        
class OrbitUtils():
    
    def __init__(self, mpc_file_in):
        """
        Get orbit information for KBMOD objects.
        
        This class is designed to use the pyOrbfit python wrappers
        of the Bernstein and Khushalani (2000) orbit fitting code to
        predict orbits for search output from KBMOD.

        Parameters
        ----------
        kbmod_info: KbmodInfo object
            The KbmodInfo object with details about the KBMOD search.
        mpc_file_in: str, default=None
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations. If None, then by default will look for
            'kbmod_mpc.dat'.
        """
        
        self.orbit = Orbit(file=mpc_file_in)
        self.kbmod_coords = KbmodInfo.mpc_reader(mpc_file_in)
        
    def set_orbit_and_kbmod_coords(self, file_in):
        """
        Set the orbit object from pyOrbfit and the observed KBMOD coordinates.
        
        Parameters
        ----------
        file_in: str
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations.
        """

        self.orbit = Orbit(file=file_in)
        self.kbmod_coords = KbmodInfo.mpc_reader(mpc_file_in)
            
    def get_orbit(self):
        """
        Get the pyOrbfit orbit object.
        
        Returns
        -------
        orbit
            pyOrbfit Orbit object
        """
        
        return self.orbit
    
    def get_kbmod_coords(self):
        """
        Get the coordinates of the kbmod observations.
        
        Returns
        -------
        kbmod_coords
            Astropy.Coordinates.SkyCoords objects
        """
        
        return self.kbmod_coords

    def predict_ephemeris(self, date_range):

        """
        Take in a time range before and after the initial observation of the object
        and predict the locations of the object along with the error parameters.

        Parameters
        ----------
        date_range: numpy array
            The dates in MJD for predicted observations.

        Returns
        -------
        pred_ra: list
            A list of predicted ra coordinates in degrees.

        pred_dec: list
            A list of predicted dec coordinates in degrees.
        """

        pos_pred_list = []

        for d in date_range-15019.5:  # Strange pyephem date conversion.
            date_start = ephem.date(d)
            pos_pred = self.orbit.predict_pos(date_start)
            pos_pred_list.append(pos_pred)

        pred_dec = []
        pred_ra = []

        for pp in pos_pred_list:
            pred_ra.append(np.degrees(pp['ra']))
            pred_dec.append(np.degrees(pp['dec']))

        return pred_ra, pred_dec

    def predict_aei_elements(self):

        """
        Predict the elements of the object

        Returns
        -------
        elements: dictionary
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: dictionary
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.
        """

        elements, errs = self.orbit.get_elements()
        
        return elements, errs
    
    def predict_abg_elements(self):

        """
        Predict the elements of the object

        Returns
        -------
        elements: dictionary
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: dictionary
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.
        """

        elements, errs = self.orbit.get_elements_abg()
        
        return elements, errs

    def predict_pixels(self, image_filename, obs_dates):

        """
        Predict the pixels locations of the object in available data.

        Parameters
        ----------
        image_filename: str
            The name of a processed image with a WCS that we can
            use to find the predicted pixel locations of the object.

        obs_dates: numpy array
            An array with times in MJD to predict the pixel locations
            of the object in the given image.

        Returns
        -------
        x_pix: numpy array
            A numpy array with the predicted x pixel locations of the object
            at the times from `obs_dates` in the given image.

        y_pix: numpy array
            A numpy array with the predicted y pixel locations of the object
            at the times from `obs_dates` in the given image.
        """
        
        new_image = fits.open(image_filename)
        new_wcs = WCS(new_image[1].header)

        pred_ra, pred_dec = self.predict_ephemeris(obs_dates)

        x_pix, y_pix = new_wcs.all_world2pix(pred_ra, pred_dec, 1)

        return x_pix, y_pix

    def plot_predictions(self, date_range, include_kbmod_obs=True):

        """
        Take in results of B&K predictions along with errors and plot predicted path
        of objects.

        Parameters
        ----------
        date_range: numpy array
            The dates in MJD for predicted observations.

        include_kbmod_obs: boolean, default=True
            If true the plot will include the observations used in the
            KBMOD search.

        Returns
        -------
        fig: matplotlib figure
            Figure object with the predicted locations for the object in
            ra, dec space color-coded by time of observation.
        """

        pred_ra, pred_dec = self.predict_ephemeris(date_range)

        fig = plt.figure()
        plt.scatter(pred_ra, pred_dec, c=date_range)
        cbar = plt.colorbar(label='mjd', orientation='horizontal',
                            pad=0.15)
        if include_kbmod_obs is True:
            plt.scatter(self.kbmod_coords.ra.deg, self.kbmod_coords.dec.deg, 
                        marker='+', s=296, edgecolors='r',
                        #facecolors='none', 
                        label='Observed Points', lw=4)
            plt.legend()
        plt.xlabel('ra')
        plt.ylabel('dec')

        return fig
    
    def plot_aei_elements_uncertainty(self, element_1, element_2, fig=None):
        
        elements, errs = self.orbit.get_elements()
        aei_idx_dict = {x: y for y, x in enumerate(elements)}
        
        el_1_val = elements[element_1]
        el_2_val = elements[element_2]
        el_1_idx = aei_idx_dict[element_1]
        el_2_idx = aei_idx_dict[element_2]
        
        dist_mean = [el_1_val, el_2_val]
        dist_covar = [[self.orbit.covar_aei[el_1_idx, el_1_idx],
                       self.orbit.covar_aei[el_1_idx, el_2_idx]],
                      [self.orbit.covar_aei[el_2_idx, el_1_idx],
                       self.orbit.covar_aei[el_2_idx, el_2_idx]]]
    
        fig = self.plot_elements_uncertainty(dist_mean, dist_covar,
                                             [element_1, element_2])
        
    def plot_elements_uncertainty(self, mean, covar, element_names, fig=None):
        
        el_dist = multivariate_normal(mean=dist_mean, cov=dist_covar)
        
        if fig is None:
            fig = plt.figure()

        sigma_contour_vals = []

        for i in range(4):
            sigma_pos = np.array([el_1_val + i*dist_covar[0][0]**.5, 
                                  el_2_val + i*dist_covar[1][1]**.5])
            pdf_val = el_dist.pdf(sigma_pos)
            sigma_contour_vals.append(pdf_val)
        sigma_contour_vals.append(0)
        print(sigma_contour_vals)
        
        
        x, y = np.mgrid[el_1_val - 5*dist_covar[0][0]**.5:el_1_val + 5*dist_covar[0][0]**.5:0.05,
                        el_2_val - 5*dist_covar[1][1]**.5:el_2_val + 5*dist_covar[1][1]**.5:0.05]
        pos = np.dstack((x, y))
        plt.contourf(x, y, el_dist.pdf(pos), levels=sigma_contour_vals[::-1], norm=mpl.colors.LogNorm(), cmap=plt.get_cmap('Reds'))
        CS=plt.contour(x, y, el_dist.pdf(pos), levels=sigma_contour_vals[::-1], colors='k')
        
        fmt = {}
        strs = [r'3 $\sigma$', r'2 $\sigma$', r'1 $\sigma$']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s
        
        plt.clabel(CS, fontsize=24, inline=1, fmt=fmt)
        #plt.colorbar()
        
        el_1_pos, el_2_pos = el_dist.rvs(10000).T
        plt.scatter(el_1_pos, el_2_pos, s=2)
        
        plt.xlabel('%s' % element_1, size=16)
        plt.ylabel('%s' % element_2, size=16)
        
        plt.xticks(size=16)
        plt.yticks(size=16)
                       
        return fig
    