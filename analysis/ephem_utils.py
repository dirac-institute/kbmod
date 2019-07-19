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

    def __init__(self, result_row, visit_min=409000, visit_max=412000):

        """
        Take in a row from the results file format and an image folder location
        and return various forms of information about the result.
        """

        self.result = result_row

        self.tract_df = pd.read_csv('visit_df.csv', index_col=0)

        visit_df = self.tract_df.query('tract_id == %i' % 
                                       self.result['tract'])
        self.visit_df = visit_df.query('visit_num > %i & '
                                       'visit_num < %i & '
                                       'filter == "g"' % (visit_min, 
                                                          visit_max))
        self.visit_df = self.visit_df.reset_index()

        self.mjd_0 = self.visit_df['image_mjd'].values[0]

        self.coords = None

    def mpc_reader(self, filename):

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

    def get_searched_radec(self, filename):

        """
        This will take an image and use its WCS to calculate the
        ra, dec locations of the object in the searched data.
        """

        hdulist = fits.open(filename)

        w = WCS(hdulist[1].header)

        zero_times = self.visit_df['image_mjd'].values - \
                     self.visit_df['image_mjd'][0]

        pix_coords_x = self.result['x0'] + \
                       self.result['x_vel']*zero_times
        pix_coords_y = self.result['y0'] + \
                       self.result['y_vel']*zero_times

        ra, dec = w.all_pix2world(pix_coords_x, pix_coords_y, 1)

        self.coords = SkyCoord(ra*u.deg, dec*u.deg)
    
    def format_results_mpc(self, fileout=None, times=None):

        """
        This method will take in a row from the results file and output the
        astrometry of the object in the searched observations into a file with
        MPC formatting.
        """

        if times is None:
            field_times = Time(self.visit_df['image_mjd'].values, format='mjd')
        else:
            field_times = Time(times, format='mjd')

        mpc_lines = []
        for t, c in zip(field_times, self.coords):
            mjd_frac = t.mjd % 1.0
            ra_hms = c.ra.hms
            dec_dms = c.dec.dms
            if dec_dms.d != 0:
                name = ("     c111112  c%4i %02i %08.5f %02i %02i %06.3f%+03i %02i %05.2f                     807" %
                        (t.datetime.year, t.datetime.month, t.datetime.day+mjd_frac,
                         ra_hms.h, ra_hms.m, ra_hms.s,
                         dec_dms.d, np.abs(dec_dms.m), np.abs(dec_dms.s)))
            else:
                if copysign(1, dec_dms.d) == -1.0:
                    dec_dms_d = '-00'
                else:
                    dec_dms_d = '+00'
                name = ("     c111112  c%4i %02i %08.5f %02i %02i %06.3f%s %02i %05.2f                     807" %
                        (t.datetime.year, t.datetime.month, t.datetime.day+mjd_frac,
                         ra_hms.h, ra_hms.m, ra_hms.s,
                         dec_dms_d, np.abs(dec_dms.m), np.abs(dec_dms.s)))

            print(name)
            mpc_lines.append(name)

        if fileout is None:
            fileout = '%i,%s,%s_mpc.dat' % (self.result['tract'],
                                            self.result['patch_horizontal'],
                                            self.result['patch_vertical'])

        with open(fileout, 'w') as f:
            for obs in mpc_lines:
                f.write(obs + '\n')

        return

    def predict_ephemeris(self, date_range, file_in=None):

        """
        Take in a time range before and after the initial observation of the object
        and predict the locations of the object along with the error parameters.
        """

        if file_in is None:
            file_in = '%i,%s,%s_mpc.dat' % (self.result['tract'],
                                            self.result['patch_horizontal'],
                                            self.result['patch_vertical'])
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
        """

        if file_in is None:
            file_in = '%i,%s,%s_mpc.dat' % (self.result['tract'],
                                            self.result['patch_horizontal'],
                                            self.result['patch_vertical'])
            o = Orbit(file=file_in)
        else:
            o = Orbit(file=file_in)
            self.coords = self.mpc_reader(file_in)

        elements, errs = o.get_elements()
        
        return elements, errs, o

    def predict_pixels(self, filename, obs_dates, file_in=None):

        """
        Predict the pixels locations of the object in available data.
        """

        hdulist = fits.open(filename)

        w = WCS(hdulist[1].header)

        pred_ra, pred_dec = self.predict_ephemeris(obs_dates, 
                                                   file_in=file_in)

        pix_coords = w.all_world2pix(pred_ra, pred_dec, 1)

        return pix_coords

    def plot_predictions(self, date_range, file_in=None, include_obs=True):

        """
        Take in results of B&K predictions along with errors and plot predicted path
        of objects.
        """

        pred_ra, pred_dec = self.predict_ephemeris(date_range, 
                                                   file_in=file_in)

        fig = plt.figure()
        plt.scatter(pred_ra, pred_dec, c=date_range)
        cbar = plt.colorbar(label='mjd', orientation='horizontal',
                            ticks=[56900, 57000, 57100, 57200],
                            pad=0.10)
        if include_obs is True:
            plt.scatter(self.coords.ra.deg, self.coords.dec.deg, 
                        marker='+', s=296, edgecolors='r',
                        #facecolors='none', 
                        label='Observed Points', lw=4)
            plt.legend()
        plt.xlabel('ra')
        plt.ylabel('dec')
        return fig