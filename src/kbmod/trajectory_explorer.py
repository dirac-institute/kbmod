import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from kbmod.results import Results
from kbmod.search import StackSearch, StampCreator, Logging
from kbmod.filters.stamp_filters import append_all_stamps, append_coadds
from kbmod.trajectory_utils import make_trajectory_from_ra_dec


logger = Logging.getLogger(__name__)


class TrajectoryExplorer:
    """A class to interactively run test trajectories through KBMOD.

    Attributes
    ----------
    config : `SearchConfiguration`
        The configuration parameters.
    search : `kb.StackSearch`
        The search object (with cached data).
    """

    def __init__(self, img_stack, config=None):
        """
        Parameters
        ----------
        im_stack : `ImageStack`
            The images to search.
        config : `SearchConfiguration`, optional
            The configuration parameters. If ``None`` uses the default
            configuration parameters.
        """
        self._data_initalized = False
        self.im_stack = img_stack
        if config is None:
            self.config = SearchConfiguration()
        else:
            self.config = config

        # Allocate and configure the StackSearch object.
        self.search = None

    def initialize_data(self):
        """Perform any needed initialization and preprocessing on the images."""
        if self._data_initalized:
            return

        # If we are using an encoded image representation on GPU, enable it and
        # set the parameters.
        if self.config["encode_num_bytes"] > 0:
            self.search.enable_gpu_encoding(self.config["encode_num_bytes"])
            logger.debug(f"Setting encoding = {self.config['encode_num_bytes']}")

        # Allocate the search structure.
        self.search = StackSearch(self.im_stack)

        self._data_initalized = True

    def evaluate_linear_trajectory(self, x, y, vx, vy):
        """Evaluate a single linear trajectory in pixel space. Skips all the filtering
        steps and returns the raw data.

        Parameters
        ----------
        x : `int`
            The starting x pixel of the trajectory.
        y : `int`
            The starting y pixel of the trajectory.
        vx : `float`
            The x velocity of the trajectory in pixels per day.
        vy : `float`
            The y velocity of the trajectory in pixels per day.

        Returns
        -------
        result : `Results`
            The results table with a single row and all the columns filled out.
        """
        self.initialize_data()

        # Evaluate the trajectory.
        trj = self.search.search_linear_trajectory(x, y, vx, vy)
        result = Results.from_trajectories([trj])

        # Get the psi and phi curves and do the sigma_g filtering.
        psi_curve = np.array([self.search.get_psi_curves(trj)])
        phi_curve = np.array([self.search.get_phi_curves(trj)])
        obs_valid = np.full(psi_curve.shape, True)
        result.add_psi_phi_data(psi_curve, phi_curve, obs_valid)

        # Get the coadds and the individual stamps.
        append_coadds(result, self.im_stack, ["sum", "mean", "median"], self.config["stamp_radius"])
        append_all_stamps(result, self.im_stack, self.config["stamp_radius"])

        return result

    def evaluate_angle_trajectory(self, ra, dec, v_ra, v_dec, wcs):
        """Evaluate a single linear trajectory in angle space. Skips all the filtering
        steps and returns the raw data.

        Parameters
        ----------
        ra : `float`
            The right ascension at time t0 (in degrees)
        dec : `float`
            The declination at time t0 (in degrees)
        v_ra : `float`
            The velocity in RA at t0 (in degrees/day)
        v_dec : `float`
            The velocity in declination at t0 (in degrees/day)
        wcs : `astropy.wcs.WCS`
            The WCS for the images.

        Returns
        -------
        result : `Results`
            The results table with a single row and all the columns filled out.
        """
        trj = make_trajectory_from_ra_dec(ra, dec, v_ra, v_dec, wcs)
        return self.evaluate_linear_trajectory(trj.x, trj.y, trj.vx, trj.vy)

    def apply_sigma_g(self, result):
        """Apply sigma G clipping to a single ResultRow. Modifies the row in-place.

        Parameters
        ----------
        result : `Results`
            A table of results to test.
        """
        clipper = SigmaGClipping(
            self.config["sigmaG_lims"][0],
            self.config["sigmaG_lims"][1],
            2,
            self.config["clip_negative"],
        )
        apply_clipped_sigma_g(clipper, result)
