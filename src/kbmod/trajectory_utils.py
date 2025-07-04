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

from kbmod.search import (
    extract_all_trajectory_vx,
    extract_all_trajectory_vy,
    extract_all_trajectory_x,
    extract_all_trajectory_y,
    Trajectory,
)


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
        The times at which to predict the positions in MJD.

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
    dt = np.asarray(times)
    # Note that we do a reassignment to avoid modifying the input, which may
    # happen if `times` is already an array and `np.asarray` is a no-op.
    zeroed_dt = dt - dt[0]

    # Predict locations in pixel space.
    x_vals = trj.x + trj.vx * zeroed_dt
    y_vals = trj.y + trj.vy * zeroed_dt

    result = wcs.pixel_to_world(x_vals, y_vals)
    return result


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
        The times of each point in zeroed days (such that first first time is zero).
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
        The times of each observed point aligned with the start time of the trajectory (in days).
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


def find_closest_trajectory(query, trj_list, times=[0.0]):
    """For a given trajectory (query) find the closest trajectory in a list.

    Parameters
    ----------
    query : `Trajectory`
        The query trajectory.
    trj_list : `list`
        The list of trajectories to search.
    times : `list`
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    result_idx : `int`
        The index of the closest matching trajectory.
    result_dist : `float`
        The average distance at the time steps.
    """
    times = np.asarray(times)
    if len(times) == 0:
        raise ValueError("Empty times array.")

    # Compute the predicted pixel locations of the query at each time.
    q_px = query.x + times * query.vx
    q_py = query.y + times * query.vy

    # For each base trajectory, compute the average distance.
    num_base = len(trj_list)
    dists = np.zeros(num_base)
    for idx, trj in enumerate(trj_list):
        dx = (trj.x + times * trj.vx) - q_px
        dy = (trj.y + times * trj.vy) - q_py
        dists[idx] = np.mean(np.sqrt(dx**2 + dy**2))

    result_idx = np.argmin(dists)
    result_dist = dists[result_idx]
    return result_idx, result_dist


def find_closest_velocity(query, trj_list):
    """For a given trajectory (query) find the trajectory
    with the closest velocity in the list.

    Parameters
    ----------
    query : `Trajectory`
        The query trajectory.
    trj_list : `list`
        The list of trajectories to search.

    Returns
    -------
    result_idx : `int`
        The index of the closest matching trajectory.
    """
    d_vx = np.array([(query.vx - trj.vx) for trj in trj_list])
    d_vy = np.array([(query.vy - trj.vy) for trj in trj_list])
    dists = np.sqrt(d_vx**2 + d_vy**2)
    return np.argmin(dists)


def trajectory_results_best_match(traj_query, results, times=[0.0]):
    """For each query trajectory find the best matching result in a Results table. This
    uses a greedy match so multiple queries can match the same result.

    Note
    ----
    This currently uses an N x M computation. If we need to handle large lists
    of query trajectories, we should look at using a spatial data structure
    like a KD-tree.

    Parameters
    ----------
    traj_query : `list`
        A list of trajectories to compare.
    results : `Results`
        The table of results with columns for x, y, vx, and vy.
    times : `list`
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    best_dist : `np.ndarray`
        A list the same length as traj_query where each entry i indicates the MSE distance
        of the best matching result row.
    best_match : `np.ndarray`
        A list the same length as traj_query where each entry i indicates the index of the
        best matching result row.
    """
    times = np.asarray(times)
    if len(times) == 0:
        raise ValueError("Empty times array.")

    # Predict the x and y positions for the base trajectories at each time (using the vectorized functions).
    base_px = predict_pixel_locations(
        times,
        results["x"],
        results["vx"],
        centered=False,
        as_int=False,
    )
    base_py = predict_pixel_locations(
        times,
        results["y"],
        results["vy"],
        centered=False,
        as_int=False,
    )

    # For each query trajectory, check if there is at least one nearby
    # base trajectory. If so, count it as matched.
    num_query = len(traj_query)
    best_dist = np.full(num_query, 0.0)
    best_match = np.full(num_query, -1)
    for idx, q_trj in enumerate(traj_query):
        # Compute the query point locations at all times.
        q_px = q_trj.x + times * q_trj.vx
        q_py = q_trj.y + times * q_trj.vy

        # Compute the average distance with each of the base predictions.
        dx = q_px[np.newaxis, :] - base_px
        dy = q_py[np.newaxis, :] - base_py
        all_dist = np.mean(np.sqrt(dx**2 + dy**2), axis=1)

        match_ind = np.argmin(all_dist)
        best_match[idx] = match_ind
        best_dist[idx] = all_dist[match_ind]

    return best_dist, best_match


def match_trajectory_sets(traj_query, traj_base, threshold, times=[0.0]):
    """Find the best matching pairs of queries (smallest distance) between the
    query trajectories and base trajectories such that each trajectory is used in
    at most one pair.

    Notes
    -----
    This function is designed to evaluate the performance of searches by determining
    which true trajectories (traj_query) were found in the result set (traj_base).

    Parameters
    ----------
    traj_query : `list`
        A list of trajectories to compare.
    traj_base : `list`
        The second list of trajectories to compare.
    threshold : float
        The distance threshold between two trajectories to count a match (in pixels).
    times : `list`
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    results : `list`
        A list the same length as traj_query where each entry i indicates the index
        of the trajectory in traj_base that best matches trajectory traj_query[i] or
        -1 if no match was found with a distance below the given threshold.
    """
    from scipy.optimize import linear_sum_assignment

    times = np.asarray(times)
    if len(times) == 0:
        raise ValueError("Empty times array.")

    if threshold <= 0.0:
        raise ValueError(f"Threshold must be greater than zero: {threshold}")

    num_query = len(traj_query)
    num_base = len(traj_base)

    # Predict the x and y positions for the base trajectories at each time (using the vectorized functions).
    base_px = predict_pixel_locations(
        times,
        extract_all_trajectory_x(traj_base),
        extract_all_trajectory_vx(traj_base),
        centered=False,
        as_int=False,
    )
    base_py = predict_pixel_locations(
        times,
        extract_all_trajectory_y(traj_base),
        extract_all_trajectory_vy(traj_base),
        centered=False,
        as_int=False,
    )

    # Compute the matrix of distances between each pair.
    dists = np.zeros((num_query, num_base))
    for q_idx, q_trj in enumerate(traj_query):
        # Compute the query point locations at all times.
        q_px = q_trj.x + times * q_trj.vx
        q_py = q_trj.y + times * q_trj.vy

        # Compute the average distance with each of the base predictions.
        dx = q_px[np.newaxis, :] - base_px
        dy = q_py[np.newaxis, :] - base_py
        dists[q_idx, :] = np.mean(np.sqrt(dx**2 + dy**2), axis=1)

    # Use scipy to solve the optimal bipartite matching problem.
    row_inds, col_inds = linear_sum_assignment(dists)

    # For each query (row) we find the best matching column and check
    # the distance against the threshold.
    results = np.full(num_query, -1)
    for row, col in zip(row_inds, col_inds):
        if dists[row, col] < threshold:
            results[row] = col

    return results
