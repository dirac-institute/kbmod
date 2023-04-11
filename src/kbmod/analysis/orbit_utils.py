from __future__ import print_function

from collections import OrderedDict
from math import copysign

import astropy.units as u
import ephem
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from pyOrbfit.Orbit import Orbit
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal


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

class OrbitUtils:
    def __init__(self, mpc_file_in):
        """
        Get orbit information for KBO objects.

        This class is designed to use the pyOrbfit python wrappers
        of the Bernstein and Khushalani (2000) orbit fitting code to
        predict orbits for KBOs.

        Parameters
        ----------
        mpc_file_in: str
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations.
        """

        self.orbit = Orbit(file=mpc_file_in)
        self.obs_coords, self.obs_times = KbmodInfo.mpc_reader(mpc_file_in)

    def set_orbit_and_obs_info(self, mpc_file_in):
        """
        Set the orbit object from pyOrbfit and the observed coordinates
        and times from an MPC formatted file.

        Parameters
        ----------
        file_in: str
            The MPC-formatted observations to use to fit the orbit and calculate
            predicted locations.
        """

        self.orbit = Orbit(file=mpc_file_in)
        self.obs_coords, self.obs_times = KbmodInfo.mpc_reader(mpc_file_in)

    def get_pyorbfit(self):
        """
        Get the pyOrbfit orbit object.

        Returns
        -------
        orbit
            pyOrbfit Orbit object
        """

        return self.orbit

    def get_obs_coords(self):
        """
        Get the coordinates of the input file observations.

        Returns
        -------
        obs_coords
            Astropy.Coordinates.SkyCoords objects
        """

        return self.obs_coords

    def get_ephemeris(self, date_range):
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

        for d in date_range - 15019.5:  # Strange pyephem date conversion.
            date_start = ephem.date(d)
            pos_pred = self.orbit.predict_pos(date_start)
            pos_pred_list.append(pos_pred)

        pred_dec = []
        pred_ra = []

        for pp in pos_pred_list:
            pred_ra.append(np.degrees(pp["ra"]))
            pred_dec.append(np.degrees(pp["dec"]))

        return pred_ra, pred_dec

    def get_pq(self, return_cov=False):
        """
        Return perihelion (p) and aphelion (q). Return errors for both
        and optionally return the covariance matrix.

        Parameters
        ----------
        return_cov: boolean, default=False
            Set to true to return the covariance matrix in addition to the
            individual errors.

        Returns
        -------
        elements: dict
            Python dictionary with perihelion and aphelion values.

        errs: dict
            Python dictionary with perihelion and aphelion.

        pq_cov: numpy ndarray, optional
            Numpy array with perihelion and aphelion covariance matrix.
        """

        elements = {}
        errs = {}

        elements["p"], errs["p"] = self.orbit.perihelion()
        elements["q"], errs["q"] = self.orbit.aphelion()

        if return_cov is True:
            cov = self.orbit.cov_pq()
            return elements, errs, cov
        else:
            return elements, errs

    def get_elements(self, basis, return_cov=False):
        """
        Predict the elements of the object

        Parameters
        ----------
        basis: str
            Specify the basis for orbital elements.
            Available options are 'aei', 'abg', 'xyz'.

        return_cov: boolean, default=False
            Set to true to return the covariance matrix in addition to the
            individual errors.

        Returns
        -------
        elements: dictionary
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: dictionary
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.

        Raises
        ------
        ValueError
            Raised when an invalid basis string is passed.
        """

        if basis == "aei":
            return self._get_aei_elements(return_cov=return_cov)
        elif basis == "abg":
            return self._get_abg_elements(return_cov=return_cov)
        elif basis == "xyz":
            return self._get_xyz_elements(return_cov=return_cov)
        else:
            raise ValueError(f"The basis {basis} is not supported.")

    def _get_aei_elements(self, return_cov=False):
        """
        Predict the elements of the object

        Parameters
        ----------
        return_cov: boolean, default=False
            Set to true to return the covariance matrix in addition to the
            individual errors.

        Returns
        -------
        elements: OrderedDict
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: OrderedDict
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.

        cov: numpy ndarray, optional
            Numpy array with perihelion and aphelion covariance matrix.
        """

        elements_aei, err_aei = self.orbit.get_elements()

        # Reorganize with OrderedDict to make sure covariance matches
        elements = OrderedDict()
        errs = OrderedDict()
        for elem_label in ["a", "e", "i", "lan", "aop", "top"]:
            elements[elem_label] = elements_aei[elem_label]
            errs[elem_label] = err_aei[elem_label]

        if return_cov is False:
            return elements, errs
        else:
            return elements, errs, self.orbit.covar_aei

    def _get_abg_elements(self, return_cov=False):
        """
        Get the elements of the object

        Parameters
        ----------
        return_cov: boolean, default=False
            Set to true to return the covariance matrix in addition to the
            individual errors.

        Returns
        -------
        elements: OrderedDict
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: OrderedDict
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.

        cov: numpy ndarray, optional
            Numpy array with perihelion and aphelion covariance matrix.
        """

        elements_abg, err_abg = self.orbit.get_elements_abg()

        # Reorganize with OrderedDict to make sure covariance matches
        elements = OrderedDict()
        errs = OrderedDict()
        for elem_label in ["a", "adot", "b", "bdot", "g", "gdot"]:
            elements[elem_label] = elements_abg[elem_label]
            errs[elem_label] = err_abg[elem_label]

        if return_cov is False:
            return elements, errs
        else:
            return elements, errs, self.orbit.covar_abg

    def _get_xyz_elements(self, return_cov=False):
        """
        Predict the barycentric elements of the object.

        Parameters
        ----------
        return_cov: boolean, default=False
            Set to true to return the covariance matrix in addition to the
            individual errors.

        Returns
        -------
        elements: OrderedDict
            A python dictionary with the orbital elements calculated from the
            Bernstein and Khushalani (2000) code.

        errs: OrderedDict
            A python dictionary with the calculated errors for the results in
            the `elements` dictionary.

        cov: numpy ndarray, optional
            Numpy array with perihelion and aphelion covariance matrix.
        """

        # Use OrderedDict to make sure covariance matches
        elements = OrderedDict()
        elements["x"] = self.orbit.orbit_xyz.x
        elements["y"] = self.orbit.orbit_xyz.y
        elements["z"] = self.orbit.orbit_xyz.z
        elements["xdot"] = self.orbit.orbit_xyz.xdot
        elements["ydot"] = self.orbit.orbit_xyz.ydot
        elements["zdot"] = self.orbit.orbit_xyz.zdot

        errs = OrderedDict()
        errs["x"] = np.sqrt(self.orbit.covar_xyz[0][0])
        errs["y"] = np.sqrt(self.orbit.covar_xyz[1][1])
        errs["z"] = np.sqrt(self.orbit.covar_xyz[2][2])
        errs["xdot"] = np.sqrt(self.orbit.covar_xyz[3][3])
        errs["ydot"] = np.sqrt(self.orbit.covar_xyz[4][4])
        errs["zdot"] = np.sqrt(self.orbit.covar_xyz[5][5])

        if return_cov is False:
            return elements, errs
        else:
            return elements, errs, self.orbit.covar_xyz

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

        pred_ra, pred_dec = self.get_ephemeris(obs_dates)

        x_pix, y_pix = new_wcs.all_world2pix(pred_ra, pred_dec, 1)

        return x_pix, y_pix

    def plot_predicted_ra_dec(self, date_range, include_kbmod_obs=True):
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

        pred_ra, pred_dec = self.get_ephemeris(date_range)

        fig = plt.figure()
        plt.scatter(pred_ra, pred_dec, c=date_range)
        cbar = plt.colorbar(label="mjd", orientation="horizontal", pad=0.15)
        if include_kbmod_obs is True:
            plt.scatter(
                self.obs_coords.ra.deg,
                self.obs_coords.dec.deg,
                marker="+",
                s=296,
                edgecolors="r",
                # facecolors='none',
                label="Observed Points",
                lw=4,
            )
            plt.legend()
        plt.xlabel("ra")
        plt.ylabel("dec")

        return fig

    def plot_elements_uncertainty(self, basis, element_1, element_2, n_samples=10000, fig=None):
        """
        Plot the orbital elements and associated 1,2,3-sigma ellipses with a number of
        samples plotted from draws of the distribution.

        Parameters
        ----------
        basis: str
            Specify the basis for orbital elements.
            Available options are 'aei', 'abg', 'xyz'.

        element_1: str
            The key in the elements dictionary for the x-axis orbital element.

        element_2: str
            The key in the elements dictionary for the y-axis orbital element.

        n_samples: int, default=10000
            Number of samples to draw from the orbital elements distribution and add to plot.

        fig: matplotlib figure, default=None
            If None it will create a new figure.

        Returns
        -------
        fig: matplotlib figure
            Figure of elements and uncertainties
        """

        elements, errs, covar = self.get_elements(basis, return_cov=True)

        idx_dict = {x: y for y, x in enumerate(elements)}

        el_1_idx = idx_dict[element_1]
        el_2_idx = idx_dict[element_2]

        # Marginalizing over a multivariate Gaussian is easy as selecting
        # only the covariance matrix elements for the desired elements
        dist_mean = np.array([elements[element_1], elements[element_2]])
        dist_covar = [
            [covar[el_1_idx][el_1_idx], covar[el_1_idx][el_2_idx]],
            [covar[el_2_idx][el_1_idx], covar[el_2_idx][el_2_idx]],
        ]
        dist_covar = np.array(dist_covar)

        fig = self._plot_elements_uncertainty(
            dist_mean, dist_covar, [element_1, element_2], n_samples=n_samples, fig=fig
        )

    def plot_pq_uncertainty(self, n_samples=10000, fig=None):
        """
        Plot the perihelion and aphelion elements and associated 1,2,3-sigma ellipses
        with a number of samples plotted from draws of the distribution.

        Parameters
        ----------
        n_samples: int, default=10000
            Number of samples to draw from the orbital elements distribution and add to plot.

        fig: matplotlib figure, default=None
            If None it will create a new figure.

        Returns
        -------
        fig: matplotlib figure
            Figure of elements and uncertainties
        """

        elements, errs, cov = self.get_pq(return_cov=True)

        fig = self._plot_elements_uncertainty(
            [elements["p"], elements["q"]], cov, ["p", "q"], n_samples=n_samples, fig=fig
        )

    def _plot_elements_uncertainty(self, dist_mean, dist_covar, element_names, n_samples=10000, fig=None):
        """
        Plot the orbital elements and associated 1,2,3-sigma ellipses with a number of
        samples plotted from draws of the distribution.

        Parameters
        ----------
        dist_mean: numpy ndarray [1 x 2]
            The mean values for the orbital elements.

        dist_covar: numpy ndarray [2 x 2]
            The covariance matrix for the two elements with axes aligned with dist_mean.

        element_names: list
            The associated names for the elements in same order as dist_mean.

        n_samples: int, default=10000
            Number of samples to draw from the orbital elements distribution and add to plot.

        fig: matplotlib figure, default=None
            If None it will create a new figure.

        Returns
        -------
        fig: matplotlib figure
            Figure of elements and uncertainties
        """

        el_dist = multivariate_normal(mean=dist_mean, cov=dist_covar)

        if fig is None:
            fig = plt.figure()

        sigma_contour_vals = []

        # Got this from here: https://commons.wikimedia.org/wiki/File:MultivariateNormal.png
        lower_chol = cholesky(dist_covar, lower=True)
        unit_circ = np.array([1, 0])
        for i in range(4):
            sigma_pos = dist_mean + np.dot(lower_chol, unit_circ * i)
            pdf_val = el_dist.pdf(sigma_pos)
            sigma_contour_vals.append(pdf_val)
        sigma_contour_vals.append(0)

        x_min = dist_mean[0] - 3.5 * dist_covar[0][0] ** 0.5
        x_max = dist_mean[0] + 3.5 * dist_covar[0][0] ** 0.5
        y_min = dist_mean[1] - 3.5 * dist_covar[1][1] ** 0.5
        y_max = dist_mean[1] + 3.5 * dist_covar[1][1] ** 0.5
        x_space = np.linspace(x_min, x_max, 100)
        y_space = np.linspace(y_min, y_max, 100)

        reds = plt.get_cmap("Reds", 256)
        red_map = reds([0, 64, 128, 256])
        new_cmap = mpl.colors.ListedColormap(red_map)
        norm = mpl.colors.BoundaryNorm(sigma_contour_vals[::-1], new_cmap.N, clip=True)

        x, y = np.meshgrid(x_space, y_space)
        pos = np.dstack((x, y))
        plt.contourf(x, y, el_dist.pdf(pos), levels=sigma_contour_vals[::-1], cmap=new_cmap, norm=norm)
        CS = plt.contour(x, y, el_dist.pdf(pos), levels=sigma_contour_vals[::-1], colors="k")

        fmt = {}
        strs = [r"3 $\sigma$", r"2 $\sigma$", r"1 $\sigma$"]
        for l, s in zip(CS.levels, strs):
            fmt[l] = s

        plt.clabel(CS, fontsize=24, inline=1, fmt=fmt)

        el_1_pos, el_2_pos = el_dist.rvs(n_samples).T
        plt.scatter(el_1_pos, el_2_pos, s=2, c="gray")

        plt.xlabel("%s" % element_names[0], size=16)
        plt.ylabel("%s" % element_names[1], size=16)

        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.gca().ticklabel_format(axis="both", style="plain")

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))

        return fig

    def plot_all_elements(self, basis, n_samples=10000, fig=None):
        """
        Plot a corner plot style figure of all the orbital elements
        in a given orbital basis and associated 1,2,3-sigma ellipses
        with a number of samples plotted from draws of the distribution for each
        pair of elements.

        Parameters
        ----------
        basis: str
            Specify the basis for orbital elements.
            Available options are 'aei', 'abg', 'xyz'.

        n_samples: int, default=10000
            Number of samples to draw from the orbital elements distribution and add to plot.

        fig: matplotlib figure, default=None
            If None it will create a new figure.

        Returns
        -------
        fig: matplotlib figure
            Figure of elements and uncertainties
        """

        elements, errs, covar = self.get_elements(basis, return_cov=True)

        if fig is None:
            fig = plt.figure()

        element_keys = list(elements.keys())
        for i in range(1, len(element_keys)):
            for idx in range(i):
                fig.add_subplot(5, 5, (i - 1) * 5 + idx + 1)
                self.plot_elements_uncertainty(
                    basis, element_keys[idx], element_keys[i], n_samples=n_samples, fig=fig
                )
