"""A collection of methods for working with Trajectories

Examples
--------
* Create a Trajectory from parameters.

* Convert a Trajectory into another data type.

* Serialize and deserialize a Trajectory.
"""

import numpy as np
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
