"""A series of helper functions for creating fake result sets, lists of 
trajectories, etc. for testing and timing."""

import numpy as np

from kbmod.results import Results
from kbmod.search import Trajectory


def make_fake_in_image_trajectory_info(num_results, height, width, dt=1.0, rng=None):
    """Create a series of fake trajectory information (x0, vx, y0, vy) such
    that the objects are within the images at all times.

    Parameters
    ----------
    num_results : `int`
        The number of results to create.
    height : `int`
        The height of the images (in pixels).
    width : `int`
        The width of the images (in pixels).
    dt : `float`
        The gap between the first and last time steps (in days).
        Default: 1.0
    rng : `numpy.random.Generator`, optional
        A random number generator to use. If None, a random generator is created.
        Default: None

    Returns
    -------
    x0 : `numpy.ndarray`
        The initial x positions of the objects.
    vx : `numpy.ndarray`
        The x velocities of the objects.
    y0 : `numpy.ndarray`
        The initial y positions of the objects.
    vy : `numpy.ndarray`
        The y velocities of the objects.
    """
    if num_results <= 0:
        raise ValueError(f"Invalid number of results {num_results}")
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid dimensions height={height}, width={width}")
    if dt <= 0.0:
        raise ValueError(f"Invalid time duration {dt}")

    # Create the random number generator if needed.
    if rng is None:
        rng = np.random.default_rng()

    # Create the fake trajectory information.
    x0 = rng.integers(0, width, size=num_results)
    xe = rng.integers(0, width, size=num_results)
    
    y0 = rng.integers(0, height, size=num_results)
    ye = rng.integers(0, height, size=num_results)

    vx = (xe - x0).astype(float) / dt
    vy = (ye - y0).astype(float) / dt

    return x0, vx, y0, vy


def make_fake_trajectories(
        num_results,
        height,
        width,
        dt=1.0,
        min_lh=0.0,
        max_lh=10.0,
        min_flux=0.0,
        max_flux=100.0,
        num_obs=10,
        rng=None,
    ):
    """Create a series of fake trajectory objects, such that the objects are within the
    images at all times.

    Parameters
    ----------
    num_results : `int`
        The number of results to create.
    height : `int`
        The height of the images (in pixels).
    width : `int`
        The width of the images (in pixels).
    dt : `float`
        The gap between the first and last time steps (in days).
        Default: 1.0
    min_flux : `float`
        The minimum flux of the objects.
        Default: 0.0
    max_flux : `float`
        The maximum flux of the objects.
        Default: 100.0
    num_obs : `int`
        The number of observations for each object.
        Default: 10
    rng : `numpy.random.Generator`, optional
        A random number generator to use. If None, a random generator is created.
        Default: None

    Returns
    -------
    trajectories : `list`
        A list of `Trajectory` objects representing the fake trajectories.
    """
    # Create the random number generator if needed.
    if rng is None:
        rng = np.random.default_rng()

    x0, vx, y0, vy = make_fake_in_image_trajectory_info(
        num_results,
        height,
        width,
        dt=dt,
        rng=rng,
    )

    # Generate the quality measures.
    flux = rng.uniform(min_flux, max_flux, size=num_results)
    lh = rng.uniform(min_lh, max_lh, size=num_results)

    # Create the fake trajectory objects.
    trajectories = []
    for i in range(num_results):
        # Create the fake trajectory object.
        trj = Trajectory(
            x = x0[i],
            y = y0[i],
            vx = vx[i],
            vy = vy[i],
            flux = flux[i],
            lh = lh[i],
            obs_count = num_obs,
        )
        trajectories.append(trj)

    return trajectories


def make_fake_results(num_times, height, width, num_results, rng=None):
    """Create a fake Results set.
    
    Parameters
    ----------
    num_times : `int`
        The number of time steps (number of images).
    height : `int`
        The height of the images (in pixels).
    width : `int`
        The width of the images (in pixels).
    num_results : `int`
        The number of result rows to create.
    rng : `numpy.random.Generator`, optional
        A random number generator to use. If None, a random generator is created.
        Default: None

    Returns
    -------
    results : `Results`
        The fake results object.
    """
    if num_times <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid dimensions num_times={num_times}, height={height}, width={width}")
    if num_results < 0:
        raise ValueError(f"Invalid number of results {num_results}")

    # Create the random number generator if needed.
    if rng is None:
        rng = np.random.default_rng()

    trjs = make_fake_trajectories(num_results, height, width, dt=float(num_times), rng=rng)
    results = Results.from_trajectories(trjs, track_filtered=False)

    # Create the fake time stamps with one observation per day starting at Jan 1, 2025.
    times = np.arange(num_times) + 60676.0
    results.set_mjd_utc_mid(times)

    return results


def add_fake_psi_phi_to_results(results, rng=None):
    """Add fake psi and phi values to the results.

    Parameters
    ----------
    results : `Results`
        The results to which to add the fake values.
    rng : `numpy.random.Generator`, optional
        A random number generator to use. If None, a random generator is created.
        Default: None

    Returns
    -------
    results : `Results`
        The results object for chaining.
    """
    # Create the random number generator if needed.
    if rng is None:
        rng = np.random.default_rng()
    num_times = results.get_num_times()
    num_results = len(results)

    # Create the fake trajectory information.
    psi = rng.normal(10.0, 0.5, size=(num_results, num_times))
    phi = rng.uniform(0.5, 1.0, size=(num_results, num_times))
    results.add_psi_phi_data(psi, phi)

    return results


def add_fake_coadds_to_results(results, coadd_name, radius, rng=None):
    """Add fake coadds to the results.

    Parameters
    ----------
    results : `Results`
        The results to which to add the fake values.
    coadd_name : `str`
        The name of the coadd to create.  The column is called "coadd_name".
    radius : `int`
        The radius of the stamps (in pixels).  The width is 2*radius+1.
    rng : `numpy.random.Generator`, optional
        A random number generator to use. If None, a random generator is created.
        Default: None

    Returns
    -------
    results : `Results`
        The results object for chaining.
    """
    if radius <= 0:
        raise ValueError(f"Invalid radius {radius}")

    # Create the random number generator if needed.
    if rng is None:
        rng = np.random.default_rng()

    num_results = len(results)
    stamp_width = 2 * radius + 1

    # Create the fake trajectory information.
    coadds = rng.uniform(0.01, 10.0, size=(num_results, stamp_width, stamp_width))
    results.table[f"coadd_{coadd_name}"] = coadds

    return results
