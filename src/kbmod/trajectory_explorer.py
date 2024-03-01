import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.filters.sigma_g_filter import apply_single_clipped_sigma_g, SigmaGClipping
from kbmod.masking import apply_mask_operations
from kbmod.result_list import ResultRow
from kbmod.search import StackSearch, StampCreator


class TrajectoryExplorer:
    """A class to interactively run test trajectories through KBMOD.

    Attributes
    ----------
    config : `SearchConfiguration`
        The configuration parameters.
    debug : `bool`
        Use verbose debug output.
    search : `kb.StackSearch`
        The search object (with cached data).
    """

    def __init__(self, img_stack, config=None, debug=False):
        """
        Parameters
        ----------
        im_stack : `ImageStack`
            The images to search.
        config : `SearchConfiguration`, optional
            The configuration parameters. If ``None`` uses the default
            configuration parameters.
        debug : `bool`
            Use verbose debug output.
        """
        self._data_initalized = False
        self.im_stack = img_stack
        if config is None:
            self.config = SearchConfiguration()
        else:
            self.config = config
        self.debug = debug

        # Allocate and configure the StackSearch object.
        self.search = None

    def initialize_data(self):
        """Perform any needed initialization and preprocessing on the images."""
        if self._data_initalized:
            return

        # Check if we need to apply legacy masking.
        if self.config["do_mask"]:
            self.im_stack = apply_mask_operations(self.config, self.im_stack)

        # If we are using an encoded image representation on GPU, enable it and
        # set the parameters.
        if self.config["encode_num_bytes"] > 0:
            self.search.enable_gpu_encoding(self.config["encode_num_bytes"])
            if self.debug:
                print(f"Setting encoding = {self.config['encode_num_bytes']}")

        # Allocate the search structure.
        self.search = StackSearch(self.im_stack)
        self.search.set_debug(self.debug)

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
        result : `ResultRow`
            The result data with all fields filled out.
        """
        self.initialize_data()

        # Evaluate the trajectory.
        trj = self.search.search_linear_trajectory(x, y, vx, vy)
        result = ResultRow(trj, self.im_stack.img_count())

        # Get the psi and phi curves and do the sigma_g filtering.
        psi_curve = np.array(self.search.get_psi_curves(trj))
        phi_curve = np.array(self.search.get_phi_curves(trj))
        result.set_psi_phi(psi_curve, phi_curve)

        # Get the individual stamps.
        stamps = StampCreator.get_stamps(self.im_stack, result.trajectory, self.config["stamp_radius"])
        result.all_stamps = np.array([stamp.image for stamp in stamps])

        return result

    def apply_sigma_g(self, result):
        """Apply sigma G clipping to a single ResultRow. Modifies the row in-place.

        Parameters
        ----------
        result : `ResultRow`
            The row to test for filtering.
        """
        clipper = SigmaGClipping(
            self.config["sigmaG_lims"][0],
            self.config["sigmaG_lims"][1],
            2,
            self.config["clip_negative"],
        )
        apply_single_clipped_sigma_g(clipper, result)
