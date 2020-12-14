from __future__ import print_function

import ephem
import os
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from math import copysign
from pyOrbfit.Orbit import Orbit
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord


class ephem_utils(object):

    """
    This class is designed to use the pyOrbfit python wrappers
    of the Bernstein and Khushalani (2000) orbit fitting code to
    predict orbits for search output from KBMOD.
    """

    def __init__(self, results_filename, image_filename,
                 visit_list, visit_mjd, results_visits, observatory):

        """
        Read in the output from a KBMOD search and store as a pandas
        DataFrame.

        Take in a list of visits and times for those visits.

        Inputs
        ------
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

    def mpc_reader(self, filename):

        """
        Read in a file with observations in MPC format and return the coordinates.

        Inputs
        ------
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

        Inputs
        ------
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
    
    def format_results_mpc(self, file_out=None):

        """
        This method will take in a row from the results file and output the
        astrometry of the object in the searched observations into a file with
        MPC formatting.

        Inputs
        ------
        file_out: str, default=None
            The output filename with the MPC-formatted observations
            of the KBMOD search result. If None, then it will save
            the output as 'kbmod_mpc.dat' and will be the default
            file in other methods below where file_in=None.
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

        if file_out is None:
            file_out = 'kbmod_mpc.dat'

        with open(file_out, 'w') as f:
            for obs in mpc_lines:
                f.write(obs + '\n')

        return(mpc_lines)

    def predict_ephemeris(self, date_range, file_in=None):

        """
        Take in a time range before and after the initial observation of the object
        and predict the locations of the object along with the error parameters.

        Inputs
        ------
        date_range: numpy array
            The dates in MJD for predicted observations.

        file_in: str, default=None
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations. If None, then by default will look for
            'kbmod_mpc.dat'.

        Returns
        -------
        pred_ra: list
            A list of predicted ra coordinates in degrees.

        pred_dec: list
            A list of predicted dec coordinates in degrees.
        """

        if file_in is None:
            file_in = 'kbmod_mpc.dat'
            o = Orbit(file=file_in)
        else:
            o = Orbit(file=file_in)
            self.coords = self.mpc_reader(file_in)

        pos_pred_list = []

        for d in date_range-15019.5:  # Strange pyephem date conversion.
            date_start = ephem.date(d)
            pos_pred = o.predict_pos(date_start)
            pos_pred_list.append(pos_pred)

        pred_dec = []
        pred_ra = []

        for pp in pos_pred_list:
            pred_ra.append(np.degrees(pp['ra']))
            pred_dec.append(np.degrees(pp['dec']))

        return pred_ra, pred_dec

    def predict_elements(self, file_in=None):

        """
        Predict the elements of the object

        Inputs
        ------
        file_in: str, default=None
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted orbital elements and associated errors.
            If None, then by default will look for 'kbmod_mpc.dat'.

        Returns
        -------
        elements: dictionary
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: dictionary
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.
        """

        if file_in is None:
            file_in = 'kbmod_mpc.dat'
            o = Orbit(file=file_in)
        else:
            o = Orbit(file=file_in)
            self.coords = self.mpc_reader(file_in)

        elements, errs = o.get_elements()
        
        return elements, errs

    def predict_pixels(self, filename, obs_dates, file_in=None):

        """
        Predict the pixels locations of the object in available data.

        Inputs
        ------
        filename: str
            The name of a processed image with a WCS that we can
            use to find the predicted pixel locations of the object.

        obs_dates: numpy array
            An array with times in MJD to predict the pixel locations
            of the object in the given image.

        file_in: str, default=None
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations. If None, then by default will look for
            'kbmod_mpc.dat'.

        Returns
        -------
        x_pix: numpy array
            A numpy array with the predicted x pixel locations of the object
            at the times from `obs_dates` in the given image.

        y_pix: numpy array
            A numpy array with the predicted y pixel locations of the object
            at the times from `obs_dates` in the given image.
        """
        
        new_image = fits.open(filename)
        new_wcs = WCS(new_image[1].header)

        pred_ra, pred_dec = self.predict_ephemeris(obs_dates, 
                                                   file_in=file_in)

        x_pix, y_pix = new_wcs.all_world2pix(pred_ra, pred_dec, 1)

        return x_pix, y_pix

    def plot_predictions(self, date_range, file_in=None, include_obs=True):

        """
        Take in results of B&K predictions along with errors and plot predicted path
        of objects.

        Inputs
        ------
        date_range: numpy array
            The dates in MJD for predicted observations.

        file_in: str, default=None
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations. If None, then by default will look for
            'kbmod_mpc.dat'.

        include_obs: boolean, default=True
            If true the plot will include the observations used in the
            KBMOD search.

        Returns
        -------
        fig: matplotlib figure
            Figure object with the predicted locations for the object in
            ra, dec space color-coded by time of observation.
        """

        pred_ra, pred_dec = self.predict_ephemeris(date_range, 
                                                   file_in=file_in)

        fig = plt.figure()
        plt.scatter(pred_ra, pred_dec, c=date_range)
        cbar = plt.colorbar(label='mjd', orientation='horizontal',
                            pad=0.15)
        if include_obs is True:
            plt.scatter(self.coords.ra.deg, self.coords.dec.deg, 
                        marker='+', s=296, edgecolors='r',
                        #facecolors='none', 
                        label='Observed Points', lw=4)
            plt.legend()
        plt.xlabel('ra')
        plt.ylabel('dec')

        return fig
