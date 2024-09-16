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
    return Trajectory(x=x0, y=y0, vx=(x1 - x0), vy=(y1 - y0))


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
