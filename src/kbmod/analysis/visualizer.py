from kbmod.analysis.plotting import plot_multiple_images
from kbmod.search import StampCreator
from .visualization_utils import *

import numpy as np
from matplotlib import pyplot as plt


class Visualizer:
    """A class for visualizing from a given `ImageStack` and `Results` set.

    Attributes
    ----------
    im_stack : `kbmod.search.ImageStack`
        `ImageStack` associated with the results.
    results : `kbmod.Results`
        The loaded `Results`.
    """

    def __init__(self, im_stack, results):
        self.im_stack = im_stack
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
            StampCreator.get_stamps(self.im_stack, trj, radius) for trj in self.trajectories
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
            valid_obstimes = []
            for i in range(len(is_valid)):
                if is_valid[i]:
                    valid_obstimes.append(self.im_stack.get_obstime(i))

            # Convert the obstimes to days and generate the number of days.
            num_days.append(len(set([mjd_to_day(t) for t in valid_obstimes])))

        # Add as a column in the results table
        self.results.table["num_days"] = num_days

    def plot_daily_coadds(self, result_idx, filename=None):
        """Plots a coadded stamp for each day of valid observations for a
        given result.

        Parameters
        ----------
        result_idx : `int`
            Index of the result to plot.
        filename : `str` or `None`
            If filename is provided, write out the plot to an
            image file.

        Raises
        ----------
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
        for i in range(self.im_stack.img_count()):
            if result_row["obs_valid"][i]:
                day = mjd_to_day(self.im_stack.get_obstime(i))
                curr_stamp = result_row["all_stamps"][i]
                # Depending on where "all_stamps" is generated may be a RawImage
                if not isinstance(curr_stamp, np.ndarray):
                    curr_stamp = curr_stamp.image

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

        plot_multiple_images(imgs, labels=labels, norm=True)

        if filename is not None:
            plt.savefig(filename)
