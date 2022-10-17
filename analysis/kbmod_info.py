from __future__ import print_function

from math import copysign

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


class KbmodInfo(object):
    """
    Hold and process the information describing the input data
    and results from a KBMOD search.
    """

    def __init__(self, results_filename, image_filename, visit_list, visit_mjd, results_visits, observatory):

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

        self.visit_df = pd.DataFrame(visit_list, columns=["visit_num"])
        self.visit_df["visit_mjd"] = visit_mjd

        results_array = np.genfromtxt(results_filename)

        # Only keep values and not property names from results file
        if len(np.shape(results_array)) == 1:
            results_proper = [results_array[1::2]]
        elif len(np.shape(results_array)) > 1:
            results_proper = results_array[:, 1::2]

        self.results_df = pd.DataFrame(
            results_proper, columns=["lh", "flux", "x0", "y0", "x_v", "y_v", "obs_count"]
        )

        image_fits = fits.open(image_filename)
        self.wcs = WCS(image_fits[1].header)

        self.results_visits = results_visits

        self.results_mjd = self.visit_df[self.visit_df["visit_num"].isin(self.results_visits)][
            "visit_mjd"
        ].values
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
        coords: astropy SkyCoord object
            A SkyCoord object with the ra, dec of the observations.
        times: astropy Time object
            Times of the observations
        """
        iso_times = []
        time_frac = []
        ra = []
        dec = []

        with open(filename, "r") as f:
            for line in f:
                year = str(line[15:19])
                month = str(line[20:22])
                day = str(line[23:25])
                iso_times.append(str("%s-%s-%s" % (year, month, day)))
                time_frac.append(str(line[25:31]))
                ra.append(str(line[32:44]))
                dec.append(str(line[44:56]))

        coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        t = Time(iso_times)
        t_obs = []
        for t_i, frac in zip(t, time_frac):
            t_obs.append(t_i.mjd + float(frac))
        obs_times = Time(t_obs, format="mjd")

        return coords, obs_times

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

        pix_coords_x = self.result["x0"] + self.result["x_v"] * zero_times
        pix_coords_y = self.result["y0"] + self.result["y_v"] * zero_times

        ra, dec = self.wcs.all_pix2world(pix_coords_x, pix_coords_y, 1)

        self.coords = SkyCoord(ra * u.deg, dec * u.deg)

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

        field_times = Time(self.results_mjd, format="mjd")

        mpc_lines = []
        for t, c in zip(field_times, self.coords):
            mjd_frac = t.mjd % 1.0
            ra_hms = c.ra.hms
            dec_dms = c.dec.dms
            if dec_dms.d != 0:
                name = (
                    "     c111112  c%4i %02i %08.5f %02i %02i %06.3f%+03i %02i %05.2f                     %s"
                    % (
                        t.datetime.year,
                        t.datetime.month,
                        t.datetime.day + mjd_frac,
                        ra_hms.h,
                        ra_hms.m,
                        ra_hms.s,
                        dec_dms.d,
                        np.abs(dec_dms.m),
                        np.abs(dec_dms.s),
                        self.obs,
                    )
                )
            else:
                if copysign(1, dec_dms.d) == -1.0:
                    dec_dms_d = "-00"
                else:
                    dec_dms_d = "+00"
                name = (
                    "     c111112  c%4i %02i %08.5f %02i %02i %06.3f%s %02i %05.2f                     %s"
                    % (
                        t.datetime.year,
                        t.datetime.month,
                        t.datetime.day + mjd_frac,
                        ra_hms.h,
                        ra_hms.m,
                        ra_hms.s,
                        dec_dms_d,
                        np.abs(dec_dms.m),
                        np.abs(dec_dms.s),
                        self.obs,
                    )
                )

            mpc_lines.append(name)

        return mpc_lines

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
        with open(file_out, "w") as f:
            for obs in mpc_lines:
                f.write(obs + "\n")

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

        pix_coords_x = self.result["x0"] + self.result["x_v"] * zero_times
        pix_coords_y = self.result["y0"] + self.result["y_v"] * zero_times

        ra, dec = self.wcs.all_pix2world(pix_coords_x, pix_coords_y, 1)

        self.coords = SkyCoord(ra * u.deg, dec * u.deg)
