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

from kbmod.search import Trajectory


def predict_pixel_locations(times, x0, vx, centered=True, as_int=True):
    """A vectorized Python implementation of the logic to predict the
    pixel locations from a starting pixel and a velocity.

    Parameters
    ----------
    times : list-like
        The length T list of zero-shifted times.
    x0 : list-like
        The length R list of starting pixel locations.
    vx : list-like
        The length R list of pixel velocities (in pixels per day) for each
        trajectory.
    centered : `bool`
        Shift the prediction to be at the center of the pixel
        (e.g. xp = x0 + vx * time + 0.5f).
        Default = True.
    as_int : `bool`
        Return the predictions as integers.
        Default = True.

    Returns
    -------
    pos : `numpy.ndarray`
        A R x T matrix where R is the number of trajectories (length of x0 and vx)
        and T is the number of times.
    """
    # Make sure everything is a numpy array.
    times = np.asarray(times)
    x0 = np.asarray(x0)
    vx = np.asarray(vx)

    # Check the arrays are the same length.
    if len(x0) != len(vx):
        raise ValueError(f"x0 and vx must be same size. Found {len(x0)} vs {len(vx)}")

    # Compute the (floating point) predicted pixel position.
    pos = vx[:, np.newaxis] * times[np.newaxis, :] + x0[:, np.newaxis]
    if centered:
        pos = pos + 0.5

    # If returned as int, we do not use an explicit floor in order to stay
    # consistent with the existing implementation.
    if as_int:
        pos = pos.astype(int)

    return pos


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


def fit_trajectory_from_pixels(x_vals, y_vals, times, centered=True):
    """Fit a linear trajectory from individual pixel values. This is not a pure best-fit
    because we restrict the starting pixels to be integers.

    Parameters
    ----------
    x_vals : `numpy.ndarray`
        The x pixel values.
    y_vals : `numpy.ndarray`
        The y pixel values.
    times : `numpy.ndarray`
        The times of each point.
    centered : `bool`
        Shift the center to start on a half pixel. Setting to ``True`` matches how
        KBMOD does the predictions during the search: x = vx * t + x0 + 0.5.
        Default: True

    Returns
    -------
    trj : `Trajectory`
        The trajectory object that best fits the observations of this fake.
    """
    num_pts = len(times)
    if len(x_vals) != num_pts or len(y_vals) != num_pts:
        raise ValueError(f"Mismatched number of points x={len(x_vals)}, y={len(x_vals)}, times={num_pts}.")
    if num_pts < 2:
        raise ValueError("At least 2 points are needed to fit a linear trajectory.")

    # Make sure the times are in sorted order.
    if num_pts > 1 and np.any(times[:-1] >= times[1:]):
        raise ValueError("Times are not in sorted order.")
    dt = times - times[0]

    # Use least squares to find the independent fits for the x and y velocities.
    T_matrix = np.vstack([dt, np.ones(num_pts)]).T
    if centered:
        vx, x0 = np.linalg.lstsq(T_matrix, x_vals - 0.5, rcond=None)[0]
        vy, y0 = np.linalg.lstsq(T_matrix, y_vals - 0.5, rcond=None)[0]
    else:
        vx, x0 = np.linalg.lstsq(T_matrix, x_vals, rcond=None)[0]
        vy, y0 = np.linalg.lstsq(T_matrix, y_vals, rcond=None)[0]

    return Trajectory(x=int(np.round(x0)), y=int(np.round(y0)), vx=vx, vy=vy)


def evaluate_trajectory_mse(trj, x_vals, y_vals, zeroed_times, centered=True):
    """Evaluate the mean squared error for the trajectory's predictions.

    Parameters
    ----------
    trj : `Trajectory`
        The trajectory object to evaluate.
    x_vals : `numpy.ndarray`
        The observed x pixel values.
    y_vals : `numpy.ndarray`
        The observed  y pixel values.
    zeroed_times : `numpy.ndarray`
        The times of each observed point aligned with the start time of the trajectory.
    centered : `bool`
        Shift the center to start on a half pixel. Setting to ``True`` matches how
        KBMOD does the predictions during the search: x = vx * t + x0 + 0.5.
        Default: True

    Returns
    -------
    mse : `float`
        The mean squared error.
    """
    num_pts = len(zeroed_times)
    if len(x_vals) != num_pts or len(y_vals) != num_pts:
        raise ValueError(f"Mismatched number of points x={len(x_vals)}, y={len(x_vals)}, times={num_pts}.")
    if num_pts == 0:
        raise ValueError("At least one point is needed to compute the error.")

    # Compute the predicted x and y values.
    pred_x = np.vectorize(trj.get_x_pos)(zeroed_times, centered=centered)
    pred_y = np.vectorize(trj.get_y_pos)(zeroed_times, centered=centered)

    # Compute the errors.
    sq_err = (x_vals - pred_x) ** 2 + (y_vals - pred_y) ** 2
    return np.mean(sq_err)
