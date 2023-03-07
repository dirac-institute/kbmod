import csv
import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from kbmod.file_utils import *


class CreateStamps(object):
    def __init__(self):
        return

    def load_lightcurves(self, lc_filename, lc_index_filename):
        """Load a set of light curves from a file.

        Parameters
        ----------
        lc_filename : str
            The filename of the lightcurves.
        lc_index_filename : str
            The filename of the good indices for the lightcurves.

        Returns
        -------
        lc : list
            A list of lightcurves.
        lc_index : list
            A list of good indices for each lightcurve.
        """
        lc = FileUtils.load_csv_to_list(lc_filename, use_dtype=float)
        lc_index = FileUtils.load_csv_to_list(lc_index_filename, use_dtype=int)
        return lc, lc_index

    def load_psi_phi(self, psi_filename, phi_filename, lc_index_filename):
        """Load the psi and phi data for each result. These are time series
        of the results' psi/phi values in each image.

        Parameters
        ----------
        psi_filename : str
            The filename of the result psi values.
        phi_filename : str
            The filename of the result phi values.
        lc_index_filename : str
            The filename of the good indices for the lightcurves.

        Returns
        -------
        psi : list
            A list of arrays containing psi values for each
            result trajctory (with one value for each image).
        phi : str
            A list of arrays containing phi values for each
            result trajctory (with one value for each image).
        lc_index : list
            A list of good indices for each lightcurve.
        """
        lc_index = FileUtils.load_csv_to_list(lc_index_filename, use_dtype=int)
        psi = FileUtils.load_csv_to_list(psi_filename, use_dtype=float)
        phi = FileUtils.load_csv_to_list(phi_filename, use_dtype=float)
        return (psi, phi, lc_index)

    def load_times(self, time_filename):
        """Load the image time stamps.

        Parameters
        ----------
        time_filename : str
            The filename of the time data.

        Returns
        -------
        times : list
            A list of times for each image.
        """
        times = FileUtils.load_csv_to_list(time_filename, use_dtype=float)
        return times

    def load_stamps(self, stamp_filename):
        """Load the stamps.

        Parameters
        ----------
        stamp_filename : str
            The filename of the stamp data.

        Returns
        -------
        stamps : list
            A list of np.arrays containing the stamps for each result.
        """
        stamps = np.genfromtxt(stamp_filename)
        if len(np.shape(stamps)) < 2:
            stamps = np.array([stamps])

        return stamps

    def max_value_stamp_filter(self, stamps, center_thresh, verbose=True):
        """Filter the stamps based on their maximum value. Keep any stamps
        where the maximum value is > center_thresh.

        Parameters
        ----------
        stamps : np array
            An np array containing the stamps for each result.
        center_thresh : float
            The filtering threshold.
        verbose : bool
            A Boolean to indicate whether to display debugging information.

        Returns
        -------
        keep_stamps : list
            An np array of stamp indices to keep.
        """
        keep_stamps = np.where(np.max(stamps, axis=1) > center_thresh)[0]
        if verbose:
            print("Center filtering keeps %i out of %i stamps." % (len(keep_stamps), len(stamps)))
        return keep_stamps

    def load_results(self, res_filename):
        """Load the result trajectories.

        Parameters
        ----------
        res_filename : str
            The filename of the results.

        Returns
        -------
        results : np array
            A np array with the result trajectories.
        """
        return FileUtils.load_results_file(res_filename)

    def plot_all_stamps(
        self,
        results,
        lc,
        lc_index,
        coadd_stamp,
        stamps,
        stamp_index=-1,
        sample=False,
        compare_SNR=False,
        show_fig=True,
    ):
        """Plot the coadded and individual stamps of the candidate object
        along with its lightcurve.
        """
        plt.rcParams.update({"figure.max_open_warning": 100})

        # Set the rows and columns for the stamp subplots.
        # These will affect the size of the lightcurve subplot.
        numCols = 5
        # Find the number of subplots to make.
        numPlots = len(stamps)
        # Compute number of rows for the plot
        numRows = numPlots // numCols
        # Add a row if numCols doesn't divide evenly into numPlots
        if numPlots % numCols:
            numRows += 1
        # Add a row if numRows=1. Avoids an error caused by ax being 1D.
        if numRows == 1:
            numRows += 1
        # Add a row for the lightcurve subplots
        numRows += 1
        if sample:
            numRows = 4
        # Plot coadd and lightcurve
        x_values = np.linspace(1, len(lc), len(lc))

        size = 21
        sigma_x = 1.4
        sigma_y = 1.4

        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)

        x, y = np.meshgrid(x, y)
        gaussian_kernel = (
            1
            / (2 * np.pi * sigma_x * sigma_y)
            * np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
        )
        sum_pipi = np.sum(gaussian_kernel**2)
        noise_kernel = np.zeros((21, 21))
        x_mask = np.logical_or(x > 5, x < -5)
        y_mask = np.logical_or(y > 5, y < -5)
        mask = np.logical_or(x_mask, y_mask)
        noise_kernel[mask] = 1
        SNR = np.zeros(len(stamps))
        signal = np.zeros(len(stamps))
        noise = np.zeros(len(stamps))

        coadd_stamp = coadd_stamp.reshape(21, 21)
        coadd_signal = np.sum(coadd_stamp * gaussian_kernel)
        coadd_noise = np.var(coadd_stamp * noise_kernel)
        coadd_SNR = coadd_signal / np.sqrt(coadd_noise * sum_pipi)
        # If SNR from kbmod is not consistent with estimated SNR, do not plot
        if compare_SNR:
            if coadd_SNR > results["lh"]:
                if results["lh"] < 0.75 * coadd_SNR:
                    return None
            elif results["lh"] > coadd_SNR:
                if coadd_SNR < 0.75 * results["lh"]:
                    return None
        # Plot the coadded stamp and the lightcurve

        # Generate the stamp plots, setting the size with figsize
        fig, ax = plt.subplots(nrows=numRows, ncols=numCols, figsize=[2 * numCols, 2 * numRows])

        # In the first row, we only want the coadd and the lightcurve.
        # Delete all other axes.
        for i in range(numCols):
            if i > 1:
                fig.delaxes(ax[0, i])

        # Turn off all axes. They will be turned back on for proper plots.
        for row in ax[1:]:
            for column in row:
                column.axis("off")
        # Plot stamps of individual visits
        ax[0, 0].imshow(coadd_stamp)
        ax[0, 1] = plt.subplot2grid((numRows, numCols), (0, 1), colspan=4, rowspan=1)
        ax[0, 1].plot(x_values, lc, "b")
        ax[0, 1].plot(x_values[lc == 0], lc[lc == 0], "g", lw=4)
        ax[0, 1].plot(x_values[lc_index], lc[lc_index], "r.", ms=15)
        ax[0, 1].xaxis.set_ticks(x_values)
        res_line = results
        title_string = (
            "Pixel (x,y) = ({:4.0f}, {:4.0f}), Vel. (x,y) = ({:8.2f},{:8.2f}), Lh = {:6.2f}, Index = {:2.0f}"
        )
        ax[0, 1].set_title(
            title_string.format(
                res_line["x"], res_line["y"], res_line["vx"], res_line["vy"], res_line["lh"], stamp_index
            )
        )
        plt.xticks(np.arange(min(x_values), max(x_values) + 1, 5.0))
        axi = 1
        axj = 0
        if sample:
            mask = np.array(random.sample(range(1, len(stamps)), 15))
        else:
            mask = np.linspace(0, len(stamps), len(stamps) + 1)
        for j, stamp in enumerate(stamps):
            stamp[np.isnan(stamp)] = 0.0
            signal[j] = np.sum(stamp * gaussian_kernel)
            noise[j] = np.var(stamp * noise_kernel)
            if noise[j] != 0:
                SNR[j] = signal[j] / np.sqrt(noise[j] * sum_pipi)
            else:
                SNR[j] = 0
            if (mask == j).any():
                im = ax[axi, axj].imshow(stamp)
                ax[axi, axj].set_title("SNR={:8.2f}".format(SNR[j]))
                ax[axi, axj].axis("on")
                # If KBMOD says the index is valid, highlight in red
                if (lc_index == j).any():
                    for axis in ["top", "bottom", "left", "right"]:
                        ax[axi, axj].spines[axis].set_linewidth(4)
                        ax[axi, axj].spines[axis].set_color("r")
                    ax[axi, axj].tick_params(axis="x", colors="red")
                    ax[axi, axj].tick_params(axis="y", colors="red")
                # Compute the axis indexes for the next iteration
                if axj < numCols - 1:
                    axj += 1
                else:
                    axj = 0
                    axi += 1
        ax[0, 0].set_title("Total SNR={:.2f}".format(coadd_SNR))
        for axis in ["top", "bottom", "left", "right"]:
            ax[0, 0].spines[axis].set_linewidth(4)
            ax[0, 0].spines[axis].set_color("r")
        ax[0, 0].tick_params(axis="x", colors="red")
        ax[0, 0].tick_params(axis="y", colors="red")
        plt.tight_layout()
        if show_fig is False:
            plt.close(fig)
        return fig

    def plot_stamps(self, results, lc, lc_index, stamps, center_thresh, fig=None):
        keep_idx = self.max_value_stamp_filter(stamps, center_thresh)

        if fig is None:
            fig = plt.figure(figsize=(12, len(lc_index) * 2))
        for i, stamp_idx in enumerate(keep_idx):
            current_lc = lc[stamp_idx]
            current_lc_index = lc_index[stamp_idx]
            x_values = np.linspace(1, len(current_lc), len(current_lc))
            fig.add_subplot(len(keep_idx), 2, (i * 2) + 1)
            plt.imshow(stamps[stamp_idx].reshape(21, 21))
            fig.add_subplot(len(keep_idx), 2, (i * 2) + 2)
            plt.plot(x_values, current_lc, "b")
            plt.plot(x_values[current_lc == 0], current_lc[current_lc == 0], "g", lw=4)
            plt.plot(x_values[current_lc_index], current_lc[current_lc_index], "r.", ms=15)
            plt.xticks(x_values)

            # Handle the case of a single result having no dimensions.
            if len(results.shape) > 0:
                res_line = results[stamp_idx]
            else:
                res_line = results
            plt.title(
                "Pixel (x,y) = (%i, %i), Vel. (x,y) = (%f, %f), Lh = %f, index = %i"
                % (res_line["x"], res_line["y"], res_line["vx"], res_line["vy"], res_line["lh"], stamp_idx)
            )
        plt.tight_layout()

        return fig

    def target_results(
        self,
        results,
        lc,
        lc_index,
        target_xy,
        stamps=None,
        center_thresh=None,
        target_vel=None,
        vel_tol=5,
        atol=10,
        title_info=None,
    ):
        keep_idx = np.linspace(0, len(lc) - 1, len(lc)).astype(int)
        if stamps is not None:
            keep_idx = self.max_value_stamp_filter(stamps, center_thresh, verbose=False)
        recovered_idx = []
        # Count the number of objects within atol of target_xy
        count = 0
        object_found = False
        for i, stamp_idx in enumerate(keep_idx):
            res_line = results[stamp_idx]
            if target_vel is not None:
                vel_truth = np.isclose(res_line["vx"], target_vel[0], atol=vel_tol) and np.isclose(
                    res_line["vy"], target_vel[1], atol=vel_tol
                )
            else:
                vel_truth = True

            if (
                np.isclose(res_line["x"], target_xy[0], atol=atol)
                and np.isclose(res_line["y"], target_xy[1], atol=atol)
                and vel_truth
            ):
                recovered_idx.append(stamp_idx)
                count += 1
        # Plot lightcurves of objects within atol of target_xy
        if count > 0:
            object_found = True
        else:
            return (0, False, [])
        y_size = count

        fig = plt.figure(figsize=(12, 2 * y_size))
        count = 0
        for i, stamp_idx in enumerate(keep_idx):
            res_line = results[stamp_idx]
            if target_vel is not None:
                vel_truth = np.isclose(res_line["vx"], target_vel[0], atol=vel_tol) and np.isclose(
                    res_line["vy"], target_vel[1], atol=vel_tol
                )
            else:
                vel_truth = True

            if (
                np.isclose(res_line["x"], target_xy[0], atol=atol)
                and np.isclose(res_line["y"], target_xy[1], atol=atol)
                and vel_truth
            ):
                current_lc = lc[stamp_idx]
                current_lc_index = lc_index[stamp_idx]
                x_values = np.linspace(1, len(current_lc), len(current_lc))
                if stamps is not None:
                    fig.add_subplot(y_size, 2, (count * 2) + 1)
                    plt.imshow(stamps[stamp_idx].reshape(21, 21))
                fig.add_subplot(y_size, 2, (count * 2) + 2)
                plt.plot(x_values, current_lc, "b")
                plt.plot(x_values[current_lc == 0], current_lc[current_lc == 0], "g.", ms=15)
                plt.plot(x_values[current_lc_index], current_lc[current_lc_index], "r.", ms=15)
                plt.xticks(x_values)
                title = "Pixel (x,y) = ({}, {}), Vel. (x,y) = ({}, {}), Lh = {}, index = {}"
                if title_info is not None:
                    title = title_info + "\n" + title
                plt.title(
                    title.format(
                        res_line["x"],
                        res_line["y"],
                        res_line["vx"],
                        res_line["vy"],
                        res_line["lh"],
                        stamp_idx,
                    )
                )
                count += 1
        plt.tight_layout()
        return (fig, object_found, recovered_idx)

    def calc_mag(self, image_files, lc, idx_list):
        flux_vals = []

        for filenum, lc_val in zip(idx_list, lc):
            hdulist = fits.open(image_files[int(filenum)])
            j_flux = lc_val / hdulist[0].header["FLUXMAG0"]
            flux_vals.append(j_flux)

        return -2.5 * np.log10(np.mean(flux_vals))


def load_stamps(results_dir, im_dir, suffix):
    image_list = sorted(os.listdir(im_dir))
    image_list = [os.path.join(im_dir, im_name) for im_name in image_list]

    stamper = CreateStamps()
    lc_filename = os.path.join(results_dir, "lc_%s.txt" % suffix)
    psi_filename = os.path.join(results_dir, "psi_{}.txt".format(suffix))
    phi_filename = os.path.join(results_dir, "phi_{}.txt".format(suffix))
    lc_index_filename = os.path.join(results_dir, "lc_index_%s.txt" % suffix)
    stamp_filename = os.path.join(results_dir, "ps_%s.txt" % suffix)
    result_filename = os.path.join(results_dir, "results_%s.txt" % suffix)

    result_exists = os.path.isfile(result_filename)

    if result_exists:
        lc_list, lc_index = stamper.load_lightcurves(lc_filename, lc_index_filename)
        psi, phi, lc_index = stamper.load_psi_phi(psi_filename, phi_filename, lc_index_filename)
        stamps = stamper.load_stamps(stamp_filename)
        all_stamps = np.load(os.path.join(results_dir, "all_ps_%s.npy" % suffix))
        results = stamper.load_results(result_filename)
        keep_idx = []
        for lc_num, lc in list(enumerate(lc_list)):
            if len(lc) > 5:
                keep_idx.append(lc_num)
        return (keep_idx, results, stamper, stamps, all_stamps, lc_list, psi, phi, lc_index)
    else:
        warnings.warn("No results found. Returning empty lists")
        return ([], [], [], [], [], [], [], [], [])
