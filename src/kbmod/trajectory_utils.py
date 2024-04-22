"""A collection of methods for working with Trajectories

Examples
--------
* Create a Trajectory from parameters.

* Convert a Trajectory into another data type.

* Serialize and deserialize a Trajectory.

* Use a trajectory and WCS to predict RA, dec positions.
"""

import numpy as np

from astropy.coordinates import SkyCoord
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
    trj.x = int(x)
    trj.y = int(y)
    trj.vx = vx
    trj.vy = vy
    trj.flux = flux
    trj.lh = lh
    trj.obs_count = int(obs_count)
    trj.valid = True
    return trj


def make_trajectory_from_ra_dec(ra, dec, v_ra, v_dec, wcs):
    """Create a trajectory object from (RA, dec) information.

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

    .. note::
       The motion is approximated as linear and will be approximately correct
       only for small temporal range and spatial region.

    Returns
    -------
    trj : `Trajectory`
        The resulting Trajectory object.
    """
    # Predict the pixel positions at t0 and t0 + 1
    x0, y0 = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    x1, y1 = wcs.world_to_pixel(SkyCoord(ra + v_ra, dec + v_dec, unit="deg"))
    return make_trajectory(x=x0, y=y0, vx=(x1 - x0), vy=(y1 - y0))


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
    if "valid" in result.dtype.names:
        trj.valid = bool(result["valid"][0])
    else:
        trj.valid = True
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
    if "valid" in trj_dict:
        trj.valid = bool(trj_dict["valid"])
    else:
        trj.valid = True
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
        "valid": trj.valid,
    }
    return dump(yaml_dict)


def update_trajectory_from_psi_phi(trj, psi_curve, phi_curve, index_valid=None, in_place=True):
    """Update the trajectory's statistic information from a psi_curve and
    phi_curve. Uses an optional index_valid mask (True/False) to mask out
    pixels.

    Parameters
    ----------
    trj : `Trajectory`
        The trajectory to update.
    psi_curve : `numpy.ndarray`
        The float psi values at each time step.
    phi_curve : `numpy.ndarray`
        The float phi values at each time step.
    index_valid : `numpy.ndarray`, optional
        An array of Booleans indicating whether the time step is valid.
    in_place : `bool`
        Update the input trajectory in-place.

    Returns
    -------
    result : `Trajectory`
        The updated trajectory. May be the same as trj if in_place=True.

    Raises
    ------
    Raises a ValueError if the input arrays are not the same size.
    """
    if len(psi_curve) != len(phi_curve):
        raise ValueError("Mismatched psi and phi curve lengths.")

    # Compute the sums of the (masked) arrays.
    if index_valid is None:
        psi_sum = np.sum(psi_curve)
        phi_sum = np.sum(phi_curve)
        num_obs = len(psi_curve)
    else:
        if len(psi_curve) != len(index_valid):
            raise ValueError("Mismatched psi/phi curve and index_valid lengths.")
        psi_sum = np.sum(psi_curve[index_valid])
        phi_sum = np.sum(phi_curve[index_valid])
        num_obs = len(psi_curve[index_valid])

    # Create a copy of the trajectory if we are not modifying in-place.
    if in_place:
        result = trj
    else:
        result = make_trajectory(x=trj.x, y=trj.y, vx=trj.vx, vy=trj.vy)

    # Update the statistics information (avoiding divide by zero).
    if phi_sum <= 0.0:
        result.lh = 0.0
        result.flux = 0.0
    else:
        result.lh = psi_sum / np.sqrt(phi_sum)
        result.flux = psi_sum / phi_sum
    result.obs_count = num_obs

    return result
