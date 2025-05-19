from matplotlib import pyplot as plt
import numpy as np

from kbmod.analysis.plotting import plot_multiple_images
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.image_utils import create_stamps_from_image_stack
from kbmod.search import ImageStack
from kbmod.util_functions import mjd_to_day


class Visualizer:
    """A class for visualizing from a given `ImageStack` and `Results` set.

    Attributes
    ----------
    im_stack : `kbmod.search.ImageStack` or `ImageStackPy`
        `ImageStack` associated with the results.
    obstimes : `np.ndarray`
        The observation times for the images in the images.
    results : `kbmod.Results`
        The loaded `Results`.
    trajectories : `list`
        List of trajectories associated with the results.
    """

    def __init__(self, im_stack, results):
        if isinstance(im_stack, ImageStackPy):
            self.obstimes = im_stack.times
        elif isinstance(im_stack, ImageStack):
            self.obstimes = np.array([im_stack.get_obstime(i) for i in range(im_stack.num_times)])
        self.results = results
        self.trajectories = results.make_trajectory_list()

    def generate_all_stamps(self, radius=10):
        """Creates a stamp cutout for each image for each
        result. Also creates a new column in the results
        table called `all_stamps`.

        Parameters
        ----------
        radius : `int`
            radius of the stamp.
        """
        self.results.table["all_stamps"] = [
            create_stamps_from_image_stack(self.im_stack, trj, radius) for trj in self.trajectories
        ]

    def count_num_days(self):
        """Counts the number of days that a given result
        has valid data in and adds a new column called
        `num_days` to the results table.
        """
        num_days = []
        for idx in range(len(self.results)):
            # Whether an observation was "valid" and included in the result
            is_valid = self.results[idx]["obs_valid"]

            # Get all of the observation times that were valid and included in the result
            valid_obstimes = self.obstimes[is_valid]

            # Convert the obstimes to days and generate the number of days.
            num_days.append(len(set([mjd_to_day(t) for t in valid_obstimes])))

        # Add as a column in the results table
        self.results.table["num_days"] = num_days

    def plot_daily_coadds(self, result_idx, filename=None, cmap=None, clim=None):
        """Plots a coadded stamp for each day of valid observations for a given result.

        Parameters
        ----------
        result_idx : `int`
            Index of the result to plot.
        filename : `str` or `None`
            If filename is provided, write out the plot to an
            image file.
        cmap : `str` or `None`
            Colormap to use for the plot.
        clim : `tuple` or `None`
            Color limits for the plot. (vmin and vmax)

        Raises
        ------
        `RuntimeError` if `num_days` or `all_stamps` are not in the
        results table (i.e. not been generated with `count_num_days` or
        `generate_all_stamps`).
        """
        if "num_days" not in self.results.table.columns:
            raise RuntimeError("`num_days` not generated, run `Visualizer.count_num_days`.")
        if "all_stamps" not in self.results.table.columns:
            raise RuntimeError("`all_stamps` not generated, run `Visualizer.generate_all_stamps`.")

        # Map each day for a result to its coadded stamp
        daily_coadds = {}
        result_row = self.results.table[result_idx]
        for i in range(self.im_stack.num_times):
            if result_row["obs_valid"][i]:
                day = mjd_to_day(self.obstimes[i])
                curr_stamp = result_row["all_stamps"][i]

                if day not in daily_coadds:
                    # Create the initial coadd
                    daily_coadds[day] = curr_stamp.copy()
                else:
                    # Add the stamps together
                    daily_coadds[day] += curr_stamp

        # First we'll plot the full coadd
        imgs = [self.results.table["stamp"][result_idx]]
        labels = [f"Coadd for result {result_idx}"]

        # Add images and labels for each individual day
        for day in daily_coadds:
            imgs.append(daily_coadds[day])
            labels.append(str(day))

        plot_multiple_images(imgs, labels=labels, norm=True, cmap=cmap, clim=clim)

        if filename is not None:
            plt.savefig(filename)
