"""A collection of methods for working with Trajectories

Examples
--------
* Create a Trajectory from parameters.

* Convert a Trajectory into another data type.

* Serialize and deserialize a Trajectory.

* Use a trajectory and WCS to predict RA, dec positions.
"""

import numpy as np

from astropy.wcs import WCS
from yaml import dump, safe_load

from kbmod.search import Trajectory


def make_trajectory(x=0, y=0, vx=0.0, vy=0.0, flux=0.0, lh=0.0, obs_count=0):
    """Create a Trajectory given the parameters with reasonable defaults.

    Parameters
    ----------
    x : `int`
        The starting x coordinate in pixels (default = 0)
    y : `int`
        The starting y coordinate in pixels (default = 0)
    vx : `float`
        The velocity in x in pixels per day (default = 0.0)
    vy : `float`
       The velocity in y in pixels per day (default = 0.0)
    flux : `float`
       The flux of the object (default = 0.0)
    lh : `float`
       The computed likelihood of the trajectory (default = 0.0)
    obs_count : `int`
       The number of observations in a trajectory (default = 0)

    Returns
    -------
    trj : `Trajectory`
        The resulting Trajectory object.
    """
    trj = Trajectory()
    trj.x = x
    trj.y = y
    trj.vx = vx
    trj.vy = vy
    trj.flux = flux
    trj.lh = lh
    trj.obs_count = obs_count
    return trj


def trajectory_predict_skypos(trj, wcs, times):
    """Predict the (RA, dec) locations of the trajectory at different times.

    Parameters
    ----------
    trj : `Trajectory`
        The corresponding trajectory object.
    wcs : `astropy.wcs.WCS`
        The WCS for the images.
    times : `list` or `numpy.ndarray`
        The times at which to predict the positions.

    .. note::
       The motion is approximated as linear and will be approximately correct
       only for small temporal range and spatial region. In essence, the new
       coordinates are calculated as:
       :math: x_new = x_old + v * (t_new - t_old)

    Returns
    -------
    result : `astropy.coordinates.SkyCoord`
        A SkyCoord with the transformed locations.
    """
    dt = np.array(times)
    dt -= dt[0]

    # Predict locations in pixel space.
    x_vals = trj.x + trj.vx * dt
    y_vals = trj.y + trj.vy * dt

    result = wcs.pixel_to_world(x_vals, y_vals)
    return result


def trajectory_from_np_object(result):
    """Transform a numpy object holding trajectory information
    into a trajectory object.

    Parameters
    ----------
    result : np object
        The result object loaded by numpy.

    Returns
    -------
    trj : `Trajectory`
        The corresponding trajectory object.
    """
    trj = Trajectory()
    trj.x = int(result["x"][0])
    trj.y = int(result["y"][0])
    trj.vx = float(result["vx"][0])
    trj.vy = float(result["vy"][0])
    trj.flux = float(result["flux"][0])
    trj.lh = float(result["lh"][0])
    trj.obs_count = int(result["num_obs"][0])
    return trj


def trajectory_from_dict(trj_dict):
    """Create a trajectory from a dictionary of the parameters.

    Parameters
    ----------
    trj_dict : `dict`
        The dictionary of parameters.

    Returns
    -------
    trj : `Trajectory`
        The corresponding trajectory object.
    """
    trj = Trajectory()
    trj.x = int(trj_dict["x"])
    trj.y = int(trj_dict["y"])
    trj.vx = float(trj_dict["vx"])
    trj.vy = float(trj_dict["vy"])
    trj.flux = float(trj_dict["flux"])
    trj.lh = float(trj_dict["lh"])
    trj.obs_count = int(trj_dict["obs_count"])
    return trj


def trajectory_from_yaml(yaml_str):
    """Parse a Trajectory object from a YAML string.

    Parameters
    ----------
    yaml_str : `str`
        The YAML string.

    Returns
    -------
    trj : `Trajectory`
        The corresponding trajectory object.
    """
    yaml_params = safe_load(yaml_str)
    trj = trajectory_from_dict(yaml_params)
    return trj


def trajectory_to_yaml(trj):
    """Serialize a Trajectory object to a YAML string.

    Parameters
    ----------
    trj : `Trajectory`
        The trajectory object to serialize.

    Returns
    -------
    yaml_str : `str`
        The YAML string.
    """
    yaml_dict = {
        "x": trj.x,
        "y": trj.y,
        "vx": trj.vx,
        "vy": trj.vy,
        "flux": trj.flux,
        "lh": trj.lh,
        "obs_count": trj.obs_count,
    }
    return dump(yaml_dict)


def predicted_trajectory_errors(trj, times, x, y):
    """Compute the distances at each time of the predicted position
    of a given trajectory and expected x and y pixel values.

    Parameters
    ----------
    trj : `Trajectory`
        The trajectory to evaluate
    times : `list` or `numpy.ndarray`
        A length N array of time steps.
    x : `list` or `numpy.ndarray`
        A length N array of the expected pixel x positions
    y : `list` or `numpy.ndarray`
        A length N array of the expected pixel y positions

    Returns
    -------
    result : `numpy.ndarray`
        The Euclidean distances at each time step.

    Raises
    ------
    Raises a ``ValueError`` if the arrays have length=0 or are
    not the same length.
    """
    if len(times) == 0:
        raise ValueError("Empty time array passed to RMS computation.")
    if len(times) != len(x) or len(x) != len(y):
        raise ValueError("Different array lengths passed in to RMS computation.")

    # Compute the predicted positions.
    zeroed_times = np.array(times) - times[0]
    xp = trj.x + trj.vx * zeroed_times
    yp = trj.y + trj.vy * zeroed_times

    # Compute the distances from the expected positions.
    return np.sqrt(np.square(xp - np.array(x)) + np.square(yp - np.array(y)))


def predicted_trajectory_rms(trj, times, x, y):
    """Compute the root mean square error of a given trajectory from
    expected x and y pixel positions.

    Parameters
    ----------
    trj : `Trajectory`
        The trajectory to evaluate
    times : `list` or `numpy.ndarray`
        A length N array of time steps.
    x : `list` or `numpy.ndarray`
        A length N array of the expected pixel x positions
    y : `list` or `numpy.ndarray`
        A length N array of the expected pixel y positions

    Returns
    -------
    result : `float`
        The root mean square error of pixel differences over all time steps.

    Raises
    ------
    Raises a ``ValueError`` if the arrays have length=0 or are
    not the same length.
    """
    dists = predicted_trajectory_errors(trj, times, x, y)
    return np.sqrt(np.sum(np.square(dists)) / len(dists))
