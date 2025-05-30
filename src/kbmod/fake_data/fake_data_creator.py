"""A class for creating fake data sets.

The FakeDataSet class allows the user to create fake data sets
for testing, including generating images with random noise and
adding artificial objects. The fake data can be saved to files
or used directly.
"""

import random
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


def image_stack_add_fake_object(stack, x, y, vx, vy, flux):
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
    flux : float
        The flux of the object.
    """
    for idx, t in enumerate(stack.zeroed_times):
        psf_kernel = stack.psfs[idx]
        psf_dim = psf_kernel.shape[0]
        psf_radius = psf_dim // 2

        px = int(x + vx * t + 0.5)
        py = int(y + vy * t + 0.5)
        for psf_y in range(psf_dim):
            for psf_x in range(psf_dim):
                img_x = px + psf_x - psf_radius
                img_y = py + psf_y - psf_radius
                if img_x >= 0 and img_x < stack.width and img_y >= 0 and img_y < stack.height:
                    stack.sci[idx][img_y, img_x] += flux * psf_kernel[psf_y, psf_x]


class FakeDataSet:
    """This class creates fake data sets for testing and demo notebooks."""

    def __init__(self, width, height, times, noise_level=2.0, psf_val=0.5, psfs=None, use_seed=False):
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
        use_seed : `bool`
            Use a deterministic seed to avoid flaky tests.
        """
        self.width = width
        self.height = height
        self.psf_val = psf_val
        self.noise_level = noise_level
        self.num_times = len(times)
        self.use_seed = use_seed
        self.trajectories = []
        self.times = times
        self.fake_wcs = None
        
        if use_seed:
            rng = np.random.default_rng(101)
        else:
            rng = np.random.default_rng()

        # Make the image stack.
        self.stack_py = make_fake_image_stack(
            height,
            width,
            times,
            noise_level=noise_level,
            psf_val=psf_val,
            psfs=psfs,
            rng=rng,
        )

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
            trj.flux,
        )

        # Save the trajectory into the internal list.
        self.trajectories.append(trj)

    def insert_random_object(self, flux):
        """Create a fake object and insert it into the image.

        Parameters
        ----------
        flux : `float`
            The flux of the object.

        Returns
        -------
        t : `Trajectory`
            The trajectory of the inserted object.
        """
        dt = self.times[-1] - self.times[0]

        # Create the random trajectory.
        t = Trajectory()
        t.x = int(random.random() * self.width)
        xe = int(random.random() * self.width)
        t.vx = (xe - t.x) / dt
        t.y = int(random.random() * self.height)
        ye = int(random.random() * self.height)
        t.vy = (ye - t.y) / dt
        t.flux = flux

        # Insert the object.
        self.insert_object(t)

        return t

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
