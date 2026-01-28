import matplotlib.pyplot as plt
import numpy as np

from kbmod.analysis.plotting import plot_image
from kbmod.core.stamp_utils import create_stamps_from_image_stack_xy
from kbmod.search import Logging
from kbmod.trajectory_utils import fit_trajectory_from_pixels, evaluate_trajectory_mse
from kbmod.util_functions import get_matched_obstimes

logger = Logging.getLogger(__name__)


class FakeInfo:
    """Information for a fake with bunch of helper functions.

    Parameters
    ----------
    times : `np.ndarray`
        The MJD times of the fakes.
    ra : `np.ndarray`
        The RA of the fakes at each time (in degrees).
    dec : `np.ndarray`
        The dec of the fakes at each time (in degrees).
    mag : `np.ndarray`
        The magnitudes of the object at each time.
    name : `str`, optional
        The name of the object.
    image_inds : `np.ndarray`
        For each observation of the fake the corresponding image index in the
        WorkUnit's image stack.
        Initially None until join_with_workunit() is called.
    x_pos_fakes : `np.ndarray`
        The x pixel position of the fakes.
        Initially None until join_with_workunit() is called.
    y_pos_fakes : `np.ndarray`
        The y pixel position of the fakes.
        Initially None until join_with_workunit() is called.
    in_image_bnds : `np.ndarray`
        A Boolean mask of whether the object's (x, y) position was in the image.
        Initially None until join_with_workunit() is called.
    trj : Trajectory
        The best fitting linear trajectory from the pixel positions.
        Initially None until join_with_workunit() is called.
    xy_stamps : `np.ndarray`
        A 3-d numpy array of stamps from the raw (x, y) values.
        Initially None until join_with_workunit() is called.
    trj_stamps : `np.ndarray`
        A 3-d numpy array of stamps from the predicted locations of the
        fitted trajectory.
        Initially None until join_with_workunit() is called.
    """

    def __init__(self, times, ra, dec, mag=None, name=None):
        self.name = name
        self.times = np.array(times)
        self.ra = np.array(ra)
        self.dec = np.array(dec)
        self.mag = np.array(mag) if mag is not None else np.zeros(len(times))
        self._validate_times()

        self.image_inds = None
        self.x_pos_fakes = None
        self.y_pos_fakes = None
        self.in_image_bnds = None
        self.trj = None
        self.xy_stamps = None
        self.trj_stamps = None

    def _validate_times(self):
        """Sort the times and remove any duplicates."""
        # Sort the data by time.
        sorted_inds = np.argsort(self.times)
        self.ra = self.ra[sorted_inds]
        self.dec = self.dec[sorted_inds]
        self.times = self.times[sorted_inds]
        self.mag = self.mag[sorted_inds]

        # Remove any duplicate times.
        if np.any(np.diff(self.times) == 0.0):
            dup_inds = np.where(np.diff(self.times) == 0.0)
            self.times = np.delete(self.times, dup_inds)
            self.ra = np.delete(self.ra, dup_inds)
            self.dec = np.delete(self.dec, dup_inds)
            self.mag = np.delete(self.mag, dup_inds)

    def __len__(self):
        return len(self.times)

    @property
    def num_times_seen(self):
        if self.in_image_bnds is None:
            raise ValueError("Must call join_with_workunit first.")
        return np.count_nonzero(self.in_image_bnds)

    def join_with_workunit(self, wu, radius=10):
        """Compute and save as much auxiliary data as we can from a WorkUnit, including:
            * The (x, y) of the fakes.
            * The best fit linear trajectory from (x, y) values.

        Parameters
        ----------
        wu : `WorkUnit`
            The WorkUnit to use
        radius : `int`
            The stamp radius to use.  If <= 0 then skip stamp generation.
        """
        obstimes = wu.get_all_obstimes()
        t0 = obstimes[0]

        # Determine in which images the fake appears.
        self.image_inds = get_matched_obstimes(obstimes, self.times, threshold=0.002)
        if np.any(self.image_inds == -1):
            raise ValueError(f"Unable to match one or more of the times:\n{obstimes}\n{self.times}")

        # Compute the (x, y) pixel locations of the fakes in each image.
        x_pos, y_pos = wu.get_pixel_coordinates(self.ra, self.dec, self.times)
        self.x_pos_fakes = x_pos
        self.y_pos_fakes = y_pos

        # Determine at which times the fake is in an image.
        in_bnds_x = (x_pos >= 0) & (x_pos < wu.im_stack.width)
        in_bnds_y = (y_pos >= 0) & (y_pos < wu.im_stack.height)
        self.in_image_bnds = in_bnds_x & in_bnds_y

        # Fit a linear trajectory to the data.
        zeroed_times = self.times - t0
        self.trj = fit_trajectory_from_pixels(x_pos, y_pos, zeroed_times)

        if radius > 0:
            num_stamps = len(self.image_inds)

            # Generate the stamps from the raw positions.
            xy_stamp_list = create_stamps_from_image_stack_xy(
                wu.im_stack,
                radius,
                x_pos.astype(int),
                y_pos.astype(int),
                list(self.image_inds),
            )

            # Generate the stamps from the fitted trajectory.
            trj_stamp_list = create_stamps_from_image_stack_xy(
                wu.im_stack,
                radius,
                (self.trj.x + self.trj.vx * zeroed_times + 0.5).astype(int),
                (self.trj.y + self.trj.vy * zeroed_times + 0.5).astype(int),
                list(self.image_inds),
            )

            # Transform to numpy arrays.
            stamp_width = 2 * radius + 1
            self.xy_stamps = np.zeros((num_stamps, stamp_width, stamp_width))
            self.trj_stamps = np.zeros((num_stamps, stamp_width, stamp_width))
            for i in range(num_stamps):
                self.xy_stamps[i, :, :] = xy_stamp_list[i].image
                self.trj_stamps[i, :, :] = trj_stamp_list[i].image

    def compute_fit_mse(self):
        """Compute the mean square error of the fitted trajectory."""
        if self.trj is None:
            raise ValueError("compute_fit_mse can only be called after join_with_workunit.")

        return evaluate_trajectory_mse(
            self.trj,
            self.x_pos_fakes,
            self.y_pos_fakes,
            self.times - self.times[0],
        )

    def compare_stamps(self, inds=None):
        """Plot pairs of raw (x, y) stamps and predicted location
        stamps for a given time step.

        Note
        ----
        Must be called after join_with_workunit().

        Parameters
        ----------
        inds : `list` or `None`
            A list of indices to use or None to use all indices.
        """
        if self.xy_stamps is None or self.trj_stamps is None:
            raise ValueError("plot_stamps can only be called after join_with_workunit.")

        if inds is None:
            inds = [i for i in range(len(self.xy_stamps))]
        num_stamps = len(inds)

        fig, axes = plt.subplots(num_stamps, 2, figsize=(3.0 * 2, 3.0 * num_stamps))
        fig.tight_layout()

        for i, index in enumerate(inds):
            plot_image(
                self.xy_stamps[index, :, :],
                ax=axes[i, 0],
                figure=fig,
                norm=True,
                title=f"Fake Given Pos\n({self.times[index]})",
                show_counts=False,
            )
            plot_image(
                self.trj_stamps[index, :, :],
                ax=axes[i, 1],
                figure=fig,
                norm=True,
                title=f"Trj Predicted Pos\n({self.times[index]})",
                show_counts=False,
            )
        plt.show()

    def plot_summary(self, figure=None, title=None):
        """Plot a summary of the fake.

        Parameters
        ----------
        figure : `matplotlib.pyplot.Figure` or `None`
            Figure, `None` by default.
        title : `str` or `None`
            The title of the figure. `None` by default.
        """
        if figure is None:
            figure = plt.figure(figsize=(9.0, 3.0), layout="constrained")

        if title is None:
            title = f"{self.name} ({len(self.times)} obs)"

        ax = figure.subplots(1, 3)

        ax[0].plot(self.times, self.ra, marker="o", color="black")
        ax[0].set_title("RA vs Time")
        ax[0].set_xlabel("Time (days)")
        ax[0].set_ylabel("RA (deg)")

        ax[1].plot(self.times, self.dec, marker="o", color="black")
        ax[1].set_title("DEC vs Time")
        ax[1].set_xlabel("Time (days)")
        ax[1].set_ylabel("DEC (deg)")

        ax[2].plot(self.times, self.mag, marker="o", color="black")
        ax[2].set_title("Mag vs Time")
        ax[2].set_xlabel("Time (days)")
        ax[2].set_ylabel("Mag")

        figure.suptitle(title)
        plt.show()


def load_fake_info_from_ecsv(filename, time_adjust=0.00112558):
    """Load the info for all the fakes from an ecsv file.

    Parameters
    ----------
    filename : `str`
        The name of the file with the fake data.
    time_adjust : `float`
        The difference between the fake's mjd_mid and the image
        time stamp (start of the image).

    Returns
    -------
    fakes : `list`
        A list of FakeInfo objects.
    """
    import pandas as pd

    logger.info(f"Loading fakes from {filename}")
    fakes_df = pd.read_csv(filename, comment="#", header=0, sep=" ")
    logger.info(f"Loaded the fakes file with {len(fakes_df)} rows.")

    obj_ids = np.unique(fakes_df["ORBITID"])
    logger.info(f"Found {len(obj_ids)} unique objects.")

    fakes = []
    for obj in obj_ids:
        row_mask = fakes_df["ORBITID"] == obj
        ra = fakes_df["RA"][row_mask].values
        dec = fakes_df["DEC"][row_mask].values
        times = fakes_df["mjd_mid"][row_mask].values + time_adjust
        mag = fakes_df["MAG"][row_mask].values
        fakes.append(FakeInfo(times, ra, dec, mag=mag, name=obj))
    return fakes
