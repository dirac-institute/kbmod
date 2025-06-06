"""A class for creating fake data sets.

The FakeDataSet class allows the user to create fake data sets
for testing, including generating images with random noise and
adding artificial objects. The fake data can be saved to files
or used directly.
"""

import numpy as np
import warnings

from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.core.psf import PSF
from kbmod.search import Trajectory
from kbmod.work_unit import WorkUnit


def create_fake_times(num_times, t0=0.0, obs_per_day=1, intra_night_gap=0.01, inter_night_gap=1):
    """Create a list of times based on a cluster of ``obs_per_day`` observations
    each night spaced out ``intra_night_gap`` within a night and ``inter_night_gap`` data between
    observing nights.

    Parameters
    ----------
    num_times : `int`
        The number of time steps (number of images).
    t0 : `float`
        The time stamp of the first observation [default=0.0]
    obs_per_day : `int`
        The number of observations on the same night [default=1]
    intra_night_gap : `float`
        The time (in days) between observations in the same night [default=0.01]
    inter_night_gap : `int`
        The number of days between observing runs [default=1]

    Returns
    -------
    result_times : `list`
        A list of ``num_times`` floats indicating the different time stamps.
    """
    if num_times <= 0:
        raise ValueError(f"Invalid number of times {num_times}")

    result_times = []
    seen_on_day = 0  # Number seen so far on the current day
    day_num = 0
    for i in range(num_times):
        result_times.append(t0 + day_num + seen_on_day * intra_night_gap)

        seen_on_day += 1
        if seen_on_day == obs_per_day:
            seen_on_day = 0
            day_num += inter_night_gap
    return result_times


def make_fake_image_stack(height, width, times, noise_level=2.0, psf_val=0.5, psfs=None, rng=None):
    """Create a fake ImageStackPy for testing.

    Parameters
    ----------
    width : int
        The width of the images in pixels.
    height : int
        The height of the images in pixels.
    times : list
        A list of time stamps.
    noise_level : float
        The level of the background noise.
        Default: 2.0
    psf_val : float
        The value of the default PSF.  Used if individual psfs are not specified.
        Default: 0.5
    psfs : `list` of `numpy.ndarray`, optional
        A list of PSF kernels. If none, Gaussian PSFs from with std=psf_val are used.
    rng : np.random.Generator
        The random number generator to use. If None creates a new random generator.
        Default: None
    """
    if rng is None:
        rng = np.random.default_rng()
    times = np.asarray(times)

    # Create the science and variance images.
    sci = [rng.normal(0.0, noise_level, (height, width)).astype(np.float32) for i in range(len(times))]
    var = [np.full((height, width), noise_level**2).astype(np.float32) for i in range(len(times))]

    # Create the PSF information.
    if psfs is None:
        psf_kernel = PSF.make_gaussian_kernel(psf_val)
        psfs = [psf_kernel for i in range(len(times))]
    elif len(psfs) != len(times):
        raise ValueError(f"The number of PSFs ({len(psfs)}) must be the same as times ({len(times)}).")

    return ImageStackPy(times, sci, var, psfs=psfs)


def image_stack_add_random_masks(stack, mask_fraction, rng=None):
    """Add random masks to the image stack.

    Parameters
    ----------
    stack : ImageStackPy
        The image stack to modify.
    mask_fraction : float
        The fraction of pixels to mask in each image [0.0, 1.0].
    rng : np.random.Generator, optional
        The random number generator to use. If None creates a new random generator.
        Default: None
    """
    if not (0.0 <= mask_fraction <= 1.0):
        raise ValueError(f"Invalid mask fraction {mask_fraction}. Must be between 0 and 1.")

    if rng is None:
        rng = np.random.default_rng()

    for idx in range(len(stack.sci)):
        mask = rng.random(stack.sci[idx].shape) < mask_fraction
        stack.sci[idx][mask] = np.nan
        stack.var[idx][mask] = np.nan


def image_stack_add_fake_object(stack, x, y, vx, vy, *, ax=0.0, ay=0.0, flux=100.0):
    """Insert a fake object given the trajectory.

    Parameters
    ----------
    stack : ImageStackPy
        The image stack to modify.
    x : int
        The x-coordinate of the object at the first time (in pixels).
    y : int
        The y-coordinate of the object at the first time (in pixels).
    vx : float
        The x-velocity of the object (in pixels per day).
    vy : float
        The y-velocity of the object (in pixels per day).
    ax : float, optional
        The x-acceleration of the object (in pixels per day^2). This is only used
        to test non-linear trajectories and defaults to 0.0.
    ay : float, optional
        The y-acceleration of the object (in pixels per day^2). This is only used
        to test non-linear trajectories and defaults to 0.0.
    flux : float, optional
        The flux of the object. Default: 100.0
    """
    for idx, t in enumerate(stack.zeroed_times):
        psf_kernel = stack.psfs[idx]
        psf_dim = psf_kernel.shape[0]
        psf_radius = psf_dim // 2

        px = int(x + vx * t + 0.5 * ax * t * t + 0.5)
        py = int(y + vy * t + 0.5 * ay * t * t + 0.5)
        for psf_y in range(psf_dim):
            for psf_x in range(psf_dim):
                img_x = px + psf_x - psf_radius
                img_y = py + psf_y - psf_radius

                # Only add flux to pixels inside the image with non-masked values.
                if (
                    img_x >= 0 and
                    img_x < stack.width and
                    img_y >= 0 and
                    img_y < stack.height and
                    np.isfinite(stack.sci[idx][img_y, img_x])
                ):
                    stack.sci[idx][img_y, img_x] += flux * psf_kernel[psf_y, psf_x]


class FakeDataSet:
    """This class creates fake data sets for testing and demo notebooks."""

    def __init__(
            self,
            width,
            height,
            times,
            *,
            mask_fraction=0.0,
            noise_level=2.0,
            psf_val=0.5, 
            psfs=None,
            artifacts_fraction=0.0,
            artifacts_mean=0.0,
            artifacts_std=2.0,
            use_seed=-1,
        ):
        """The constructor.

        Parameters
        ----------
        width : `int`
            The width of the images in pixels.
        height : `int`
            The height of the images in pixels.
        times : `list`
            A list of time stamps, such as produced by ``create_fake_times``.
        noise_level : `float`
            The level of the background noise.
            Default: 2.0
        psf_val : `float`
            The value of the default PSF std.  Used if individual psfs are not specified.
            Default: 0.5
        psfs : `list` of `numpy.ndarray`, optional
            A list of PSF kernels. If none, Gaussian PSFs from with std=psf_val are used.
        mask_fraction : `float`, optional
            The fraction of pixels to mask in each image [0.0, 1.0].
            Default: 0.0 (no masks).
        artifacts_fraction : `float`, optional
            The fraction of pixels to modify with noise artifacts that are brighter
            than the background noise [0.0, 1.0].
            Default: 0.0 (no artifacts).
        artifacts_mean : `float`, optional
            The mean value of the artifacts in units of flux.
            Default: 0.0 (no artifacts).
        artifacts_std : `float`, optional
            The standard deviation of the artifacts.
            Default: 2.0 (same as noise level).
        use_seed : `int`
            Use a deterministic seed to avoid flaky tests.
            Default: -1 (no seed, random behavior).
        """
        self.times = times
        self.num_times = len(times)
        if self.num_times == 0:
            raise ValueError("The list of times must not be empty.")

        # Base image information.
        self.width = width
        self.height = height
        self.noise_level = noise_level
        self.mask_fraction = mask_fraction
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: width={width}, height={height}")
        if not (0.0 <= mask_fraction <= 1.0):
            raise ValueError(f"Invalid mask fraction {mask_fraction}. Must be between 0 and 1.")

        # Artifacts information.
        self.artifacts_fraction = artifacts_fraction
        self.artifacts_mean = artifacts_mean
        self.artifacts_std = artifacts_std
        if not (0.0 <= artifacts_fraction <= 1.0):
            raise ValueError(f"Invalid artifacts fraction {artifacts_fraction}. Must be between 0 and 1.")

        # PSF information.
        self.psf_val = psf_val
        self.psfs = psfs

        # Set up the random number generator.
        self.use_seed = use_seed
        if use_seed < 0:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(use_seed)

        # Set the auxiliary data to empty.
        self.trajectories = []
        self.fake_wcs = None

        # Make the image stack, mask out the pixels, and add the artifacts.
        self.reset()

    def reset(self):
        """Regenerate the image stack and clear the inserted objects."""
        self.stack_py = make_fake_image_stack(
            self.height,
            self.width,
            self.times,
            noise_level=self.noise_level,
            psf_val=self.psf_val,
            psfs=self.psfs,
            rng=self.rng,
        )

        if self.mask_fraction > 0.0:
            image_stack_add_random_masks(self.stack_py, self.mask_fraction, rng=self.rng)
        if self.artifacts_fraction > 0.0:
            self.insert_random_artifacts(
                self.artifacts_fraction,
                self.artifacts_mean,
                self.artifacts_std,
            )

        # Clear the list of inserted objects and WCS.
        self.trajectories = []

    def set_wcs(self, new_wcs):
        """Set a new fake WCS to be used for this data.

        Parameters
        ----------
        new_wcs : `astropy.wcs.WCS`
            The WCS to use.
        """
        self.fake_wcs = new_wcs

    def insert_object(self, trj):
        """Insert a fake object given the trajectory.

        Parameters
        ----------
        trj : `Trajectory`
            The trajectory of the fake object to insert.
        """
        # Insert the object into the images.
        image_stack_add_fake_object(
            self.stack_py,
            trj.x,
            trj.y,
            trj.vx,
            trj.vy,
            flux=trj.flux,
        )

        # Save the trajectory into the internal list.
        self.trajectories.append(trj)

    def trajectory_is_within_bounds(self, trj):
        """Check if the trajectory is within the image bounds for all times."""
        dt = self.times[-1] - self.times[0]
        xe = trj.x + trj.vx * dt
        ye = trj.y + trj.vy * dt
        return (
            0 <= trj.x < self.width and
            0 <= trj.y < self.height and
            0 <= xe < self.width and
            0 <= ye < self.height
        )

    def insert_random_object(self, flux, vx=None, vy=None):
        """Create a fake object that will be within the image bounds
        in all images.

        Parameters
        ----------
        flux : `float`
            The flux of the object.
        vx : `float` or `list` of `float`, optional
            The x-velocity or list of allowable x-velocities (in pixels per day).
            If None, a random x-velocity is generated.
        vy : `float` or `list` of `float`, optional
            The y-velocity or list of allowable y-velocities (in pixels per day).
            If None, a random y-velocity is generated.

        Returns
        -------
        trj : `Trajectory`
            The trajectory of the inserted object.
        """
        dt = self.times[-1] - self.times[0]

        trj = Trajectory(flux=flux)
        is_valid = False
        itr = 0
        while not is_valid and itr < 1000:
            # We use rejection sampling to ensure the object is visible in all images.
            trj.x = int(self.rng.random() * self.width)
            trj.y = int(self.rng.random() * self.height)

            if vx is None:
                # If no x-velocity is specified, create one by picking a random end point.
                xe = int(self.rng.random() * self.width)
                trj.vx = (xe - trj.x) / dt
                print(f"No vx given: {xe}, {trj.vx}")
            elif np.isscalar(vx):
                # If a scalar is given, use it as the x-velocity.
                trj.vx = vx
                print(f"One vx given: {trj.vx}")
            else:
                # If a vector is given, pick a random one.
                trj.vx = self.rng.choice(vx)
                print(f"Many vx given: {trj.vx}")

            if vy is None:
                # If no y-velocity is specified, create one by picking a random end point.
                ye = int(self.rng.random() * self.height)
                trj.vy = (ye - trj.y) / dt
                print(f"No vy given: {ye}, {trj.vy}")
            elif np.isscalar(vy):
                # If a scalar is given, use it as the y-velocity.
                trj.vy = vy
                print(f"One vy given: {trj.vy}")
            else:
                # If a vector is given, pick a random one.
                trj.vy = self.rng.choice(vy)
                print(f"Many vy given: {trj.vy}")

            # Check if the object is visible in all images.  If not, try again.
            is_valid = self.trajectory_is_within_bounds(trj)
            if not is_valid:
                raise ValueError("What?")
            itr += 1
        
        if not is_valid:
            warnings.warn(
                "Failed to create a valid random object after 1000 attempts. "
                "The object may not be visible in all images."
            )
        
        # Insert the object.
        self.insert_object(trj)

        return trj

    def insert_random_objects_from_generator(self, num_trj, generator, flux):
        """Insert a number of random objects based on a given TrajectorGenerator.
        This ensures that the inserted objects can be recovered by the search. 

        Parameters
        ----------
        num_trj : `int`
            The number of trajectories to insert.
        generator : `TrajectoryGenerator`
            The generator to use for creating the trajectories.
        flux : `float`
            The flux of the object.

        Returns
        -------
        trjs : `List` of `Trajectory`
            The list of all the trajectories inserted.
        """
        # Extract the list of possible velocities from the generator.
        candidate_trjs = [trj for trj in generator]
        if len(candidate_trjs) == 0:
            raise ValueError("The generator did not produce any trajectories.")

        # Insert the objects one at a time, using rejection sampling to ensure
        # they appear in all images.
        trjs = []
        itr = 0
        while len(trjs) < num_trj and itr < 10 * num_trj:
            # Pick a random trajectory from the generator to use for its velocity.
            trj_v = self.rng.choice(candidate_trjs)

            trj = Trajectory(
                x = int(self.rng.random() * self.width),
                y = int(self.rng.random() * self.height),
                vx = trj_v.vx,
                vy = trj_v.vy,
                flux=flux,
            )
            if self.trajectory_is_within_bounds(trj):
                # If the trajectory is valid, insert it.
                self.insert_object(trj)
                trjs.append(trj)

            itr += 1
            
        if len(trjs) < num_trj:
            warnings.warn(f"Only inserted {len(trjs)} out of {num_trj} requested trajectories.")

        return trjs

    def insert_random_artifacts(self, fraction, mean, std):
        """Insert noise artifacts into the images that are brighter
        than the background noise.

        Parameters
        ----------
        fraction : `float`
            The fraction of pixels to modify [0.0, 1.0]
        mean : `float`
            The mean value of the artifacts in units of flux.
        std : `float`
            The standard deviation of the artifacts.
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"Invalid fraction {fraction}. Must be between 0 and 1.")

        for idx in range(len(self.stack_py.sci)):
            to_add = self.rng.random(self.stack_py.sci[idx].shape) < fraction

            # Only add to unmasked pixels.
            mask = self.stack_py.get_mask(idx)
            to_add &= ~mask

            self.stack_py.sci[idx][to_add] += self.rng.normal(mean, std, size=np.sum(to_add))

    def get_work_unit(self, config=None):
        """Create a WorkUnit from the fake data.
        
        Parameters
        ----------
        filename : `str`
            The name of the resulting WorkUnit file.
        config : `SearchConfiguration`, optional
            The configuration parameters to use. If None then uses
            default values.
        """
        if config is None:
            config = SearchConfiguration()
        
        # Create the WorkUnit, but disable warnings about no WCS since this if fake data.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            work = WorkUnit(im_stack=self.stack_py, config=config, wcs=self.fake_wcs)
        return work
    
    def save_fake_data_to_work_unit(self, filename, config=None):
        """Create the fake data in a WorkUnit file.

        Parameters
        ----------
        filename : `str`
            The name of the resulting WorkUnit file.
        config : `SearchConfiguration`, optional
            The configuration parameters to use. If None then uses
            default values.
        """
        work = self.get_work_unit(config)
        work.to_fits(filename)
