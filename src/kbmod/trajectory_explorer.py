import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.configuration import SearchConfiguration
from kbmod.filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from kbmod.results import Results
from kbmod.run_search import configure_kb_search_stack
from kbmod.search import DebugTimer, StackSearch, Logging, Trajectory
from kbmod.filters.clustering_filters import NNSweepFilter
from kbmod.filters.stamp_filters import append_all_stamps, append_coadds
from kbmod.trajectory_generator import PencilSearch, VelocityGridSearch
from kbmod.trajectory_utils import make_trajectory_from_ra_dec


logger = Logging.getLogger(__name__)


class TrajectoryExplorer:
    """A class to interactively run test trajectories through KBMOD.

    Attributes
    ----------
    clipper : `SigmaGClipping`
        The sigma-G clipping object.
    config : `SearchConfiguration`
        The configuration parameters.
    im_stack : `ImageStackPy`
        The images to search.
    preload_data : `bool`
        If ``True`` preload all the image data into memory.
    search : `kb.StackSearch`
        The search object (with cached data).
    """

    def __init__(self, im_stack, config=None, preload_data=False):
        """
        Parameters
        ----------
        im_stack : `ImageStackPy`
            The images to search.
        config : `SearchConfiguration`, optional
            The configuration parameters. If ``None`` uses the default
            configuration parameters.
        preload_data : `bool`, optional
            If ``True`` preload all the image data into memory. Default is ``False``.
        """
        self._data_initalized = False
        self.im_stack = im_stack
        if config is None:
            self.config = SearchConfiguration()
        else:
            self.config = config
        self.preload_data = preload_data

        # Set up the clipped sigma-G filter.
        self.clipper = SigmaGClipping(
            self.config["sigmaG_lims"][0],
            self.config["sigmaG_lims"][1],
            2,
            self.config["clip_negative"],
        )

        # Allocate and configure the StackSearch object.
        self.search = None

    def initialize_data(self, config=None):
        """Initialize the data, including applying the configuration parameters.

        Parameters
        ----------
        config : `SearchConfiguration`, optional
            Any custom configuration parameters to use for this run.
            If ``None`` uses the default configuration parameters.
        """
        if config is None:
            config = self.config

        if self._data_initalized:
            # Always reapply the configuration parameters (except image encoding)
            # in case we used different parameters on a previous search.
            configure_kb_search_stack(self.search, config)

            # Nothing else to do
            return

        # Allocate the search structure.
        if isinstance(self.im_stack, ImageStackPy):
            self.search = StackSearch(
                self.im_stack.sci,
                self.im_stack.var,
                self.im_stack.psfs,
                self.im_stack.zeroed_times,
                self.config["encode_num_bytes"],
            )
        else:
            raise TypeError("Unsupported image stack type.")

        configure_kb_search_stack(self.search, config)
        if self.preload_data:
            self.search.preload_psi_phi_array()

        self._data_initalized = True

    def evaluate_linear_trajectory(self, x, y, vx, vy, use_kernel):
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
        use_kernel : `bool`
            Force the use of the exact kernel code (including on GPU-sigma G).

        Returns
        -------
        result : `Results`
            The results table with a single row and all the columns filled out.
        """
        self.initialize_data()

        # Evaluate the trajectory.
        trj = self.search.search_linear_trajectory(x, y, vx, vy, use_kernel)
        result = Results.from_trajectories([trj])

        # Get the psi and phi curves and do the sigma_g filtering.
        num_times = self.im_stack.num_times
        psi_phi = self.search.get_all_psi_phi_curves([trj])
        psi_curve = psi_phi[:, :num_times]
        phi_curve = psi_phi[:, num_times:]
        obs_valid = np.full(psi_curve.shape, True, dtype=bool)
        result.add_psi_phi_data(psi_curve, phi_curve, obs_valid)

        # Get the coadds and the individual stamps.
        append_coadds(result, self.im_stack, ["sum", "mean", "median"], self.config["stamp_radius"])
        append_all_stamps(result, self.im_stack, self.config["stamp_radius"])

        # Compute the clipped sigma-G filtering, but save it as another column.
        lh = result.compute_likelihood_curves(filter_obs=True, mask_value=np.nan)
        obs_valid = self.clipper.compute_clipped_sigma_g_matrix(lh)
        result.table["sigma_g_res"] = obs_valid

        return result

    def evaluate_angle_trajectory(self, ra, dec, v_ra, v_dec, wcs, use_kernel):
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
        use_kernel : `bool`
            Force the use of the exact kernel code (including on GPU-sigma G).

        Returns
        -------
        result : `Results`
            The results table with a single row and all the columns filled out.
        """
        trj = make_trajectory_from_ra_dec(ra, dec, v_ra, v_dec, wcs)
        return self.evaluate_linear_trajectory(trj.x, trj.y, trj.vx, trj.vy, use_kernel)

    def evaluate_around_linear_trajectory(
        self,
        x,
        y,
        vx,
        vy,
        pixel_radius=5,
        max_ang_offset=0.2618,
        ang_step=0.035,
        max_vel_offset=10.0,
        vel_step=0.5,
        use_gpu=True,
    ):
        """Evaluate all the trajectories within a local neighborhood of the given trajectory.
        No filtering is done at all.

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
        pixel_radius : `int`
            The number of pixels to evaluate to each side of the Trajectory's starting pixel.
        max_ang_offset : `float`
            The maximum offset of a candidate trajectory from the original (in radians)
        ang_step : `float`
            The step size to explore for each angle (in radians)
        max_vel_offset : `float`
            The maximum offset of the velocity's magnitude from the original (in pixels per day)
        vel_step : `float`
            The step size to explore for each velocity magnitude (in pixels per day)
        use_gpu : `bool`
            Run the search on GPU.

        Returns
        -------
        result : `Results`
            The results table with a single row and all the columns filled out.
        """
        if pixel_radius < 0:
            raise ValueError(f"Pixel radius must be >= 0. Got {pixel_radius}")
        num_pixels = (2 * pixel_radius + 1) * (2 * pixel_radius + 1)
        logger.debug(f"Testing {num_pixels} starting pixels.")

        # Create a pencil search around the given trajectory.
        trj_generator = PencilSearch(vx, vy, max_ang_offset, ang_step, max_vel_offset, vel_step)
        num_trj = len(trj_generator)
        logger.debug(f"Exploring {num_trj} trajectories per starting pixel.")

        # Set the search bounds to right around the trajectory's starting position and
        # turn off all filtering.
        reduced_config = self.config.copy()
        reduced_config.set("x_pixel_bounds", [x - pixel_radius, x + pixel_radius + 1])
        reduced_config.set("y_pixel_bounds", [y - pixel_radius, y + pixel_radius + 1])
        reduced_config.set("results_per_pixel", min(num_trj, 10_000))
        reduced_config.set("gpu_filter", False)
        reduced_config.set("num_obs", 1)
        reduced_config.set("max_lh", 1e25)
        reduced_config.set("lh_level", -1e25)
        self.initialize_data(config=reduced_config)

        # Do the actual search.
        search_timer = DebugTimer("grid search", logger)
        candidates = [trj for trj in trj_generator]
        self.search.search_all(candidates, use_gpu)
        search_timer.stop()

        # Load all of the results without any filtering.
        logger.debug(f"Loading {num_pixels * num_trj} results.")
        trjs = self.search.get_results(0, num_pixels * num_trj)
        results = Results.from_trajectories(trjs)

        return results

    def refine_linear_trajectory(
        self,
        x,
        y,
        vx,
        vy,
        *,
        pixel_radius=50,
        max_dv=10.0,
        dv_steps=21,
        max_results=1,
        use_gpu=True,
    ):
        """Evaluate all trajectories within a local neighborhood of the given trajectory
        (applying standard filtering) and return the best one.

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
        pixel_radius : `int`
            The number of pixels to evaluate to each side of the Trajectory's starting pixel.
        max_dv : `float`
            The maximum change in per-coordinate pixel velocity to explore (in pixels/day)
        dv_steps: `int`
            The number of steps to explore in each velocity dimension.
        max_results : `int`
            The maximum number of results to return.
        use_gpu : `bool`
            Run the search on GPU.

        Returns
        -------
        result : `Results`
            The results table with a single row and all the columns filled out.
        """
        if pixel_radius < 0:
            raise ValueError(f"Pixel radius must be >= 0. Got {pixel_radius}")
        num_pixels = (2 * pixel_radius + 1) * (2 * pixel_radius + 1)
        logger.debug(f"Testing {num_pixels} starting pixels.")

        # Create a pencil search around the given trajectory.
        if max_dv < 0 or dv_steps < 1:
            raise ValueError(f"max_dv must be >= 0 and dv_steps must be >= 1.")

        trj_generator = VelocityGridSearch(
            dv_steps,
            vx - max_dv,
            vx + max_dv,
            dv_steps,
            vy - max_dv,
            vy + max_dv,
        )
        candidates = [trj for trj in trj_generator]
        logger.debug(f"Exploring {len(candidates)} trajectories per starting pixel.")

        # Set the search bounds to right around the trajectory's starting position.
        reduced_config = self.config.copy()
        reduced_config.set("x_pixel_bounds", [x - pixel_radius, x + pixel_radius + 1])
        reduced_config.set("y_pixel_bounds", [y - pixel_radius, y + pixel_radius + 1])
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1. Got {max_results}")
        reduced_config.set("results_per_pixel", max_results)
        self.initialize_data(config=reduced_config)

        # Do the actual search.
        search_timer = DebugTimer("grid search", logger)
        self.search.search_all(candidates, use_gpu)
        search_timer.stop()

        # Load all of the results without any filtering.
        logger.debug(f"Loading {max_results} results.")
        trjs = self.search.get_results(0, max_results)
        results = Results.from_trajectories(trjs)

        return results

    def apply_sigma_g(self, result):
        """Apply sigma G clipping to a single ResultRow. Modifies the row in-place.

        Parameters
        ----------
        result : `Results`
            A table of results to test.
        """
        apply_clipped_sigma_g(self.clipper, result)


def refine_all_results(
    results,
    im_stack,
    config,
    *,
    deduplicate=True,
    pixel_radius=50,
    max_dv=10.0,
    dv_steps=21,
):
    """Refine the trajectories in results by re-running the search in a small
    neighborhood around each trajectory.

    Parameters
    ----------
    results : `Results`
        The current table of results including the per-pixel trajectories.
        This is modified in-place.
    im_stack : `ImageStackPy`
        The stack of image data.
    config : `SearchConfiguration`
        The configuration parameters
    deduplicate : `bool`, optional
        If True, remove duplicate trajectories after refinement (this includes
        any trajectories whose start and end points are within the pixel radius).
    pixel_radius : `int`
        The number of pixels to evaluate to each side of the Trajectory's starting pixel.
    max_dv : `float`
        The maximum change in per-coordinate pixel velocity to explore (in pixels/day)
    dv_steps: `int`
        The number of steps to explore in each velocity dimension.

    Returns
    -------
    refined : `Results`
        A new Results object with the refined trajectories.
    """
    # Get the information from the result table.
    num_res = len(results)
    if num_res == 0:
        return results  # Nothing to do

    new_trjs = []
    trj_explorer = TrajectoryExplorer(im_stack, config=config, preload_data=True)
    for idx in range(num_res):
        refined = trj_explorer.refine_linear_trajectory(
            results["x"][idx],
            results["y"][idx],
            results["vx"][idx],
            results["vy"][idx],
            pixel_radius=pixel_radius,
            max_dv=max_dv,
            dv_steps=dv_steps,
            max_results=1,
        )
        best_trj = Trajectory(
            x=refined["x"][0],
            y=refined["y"][0],
            vx=refined["vx"][0],
            vy=refined["vy"][0],
            flux=refined["flux"][0],
            lh=refined["likelihood"][0],
            obs_count=refined["obs_count"][0],
        )
        new_trjs.append(best_trj)

    # Create and sort the results. Preserve the UUID if present.
    new_results = Results.from_trajectories(new_trjs)
    if "uuid" in results.colnames:
        new_results.table["uuid"] = results["uuid"]
    new_results.sort("likelihood", descending=True)

    # Deduplicate the results (if needed). We keep results that are the best
    # in their neighborhood at either the starting or ending point.
    if deduplicate:
        zeroed_times = im_stack.zeroed_times
        keep_t0 = NNSweepFilter(pixel_radius, [0.0]).keep_indices(new_results)
        keep_tL = NNSweepFilter(pixel_radius, [zeroed_times[-1]]).keep_indices(new_results)
        keep_inds = np.union1d(keep_t0, keep_tL)
        new_results.filter_rows(keep_inds, "deduplicate")

    return new_results
