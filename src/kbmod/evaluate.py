import math

from kbmod.search import *


def ave_trajectory_distance(trjA, trjB, times=[0.0]):
    """
    Evaluate the average distance between two trajectories (in pixels)
    at different times.

    Arguments:
        trjA : trajectory
            The first trajectory to evaluate.
        trjB : trajectory
            The second trajectory to evaluate.
        times : list
            The list of zero-shifted times at which to evaluate the
            matches. The average of the distances at these times
            are used.

    Returns:
        float : The average distance in pixels.
    """
    num_times = len(times)
    assert num_times > 0

    posA = [compute_traj_pos(trjA, times[i]) for i in range(num_times)]
    posB = [compute_traj_pos(trjB, times[i]) for i in range(num_times)]
    ave_dist = ave_trajectory_dist(posA, posB)
    return ave_dist


def find_unique_overlap(traj_query, traj_base, threshold, times=[0.0]):
    """
    Finds the set of trajectories in traj_query that are 'close' to
    trajectories in traj_base such that each trajectory in traj_base
    is used at most once.

    Used to evaluate the performance of algorithms.

    Arguments:
        traj1 : list
            A list of trajectories to compare.
        traj2 : list
            The second list of trajectories to compare.
        threshold : float
            The distance threshold between two observations
            to count a match (in pixels).
        times : list
            The list of zero-shifted times at which to evaluate the
            matches. The average of the distances at these times
            are used.

    Returns:
        list : a list of trajectories that appear in both traj1 and traj2
               where each trajectory in each set is only used once.
    """
    num_times = len(times)
    size_base = len(traj_base)
    used = [False] * size_base

    results = []
    for query in traj_query:
        best_dist = 10.0 * threshold
        best_ind = -1

        # Check the current query against all unused base trajectories.
        for j in range(size_base):
            if not used[j]:
                dist = ave_trajectory_distance(query, traj_base[j], times)
                if dist < best_dist:
                    best_dist = dist
                    best_ind = j

        # If we found a good match, save it.
        if best_dist <= threshold:
            results.append(query)
            used[best_ind] = True
    return results


def find_set_difference(traj_query, traj_base, threshold, times=[0.0]):
    """
    Finds the set of trajectories in traj_query that are NOT 'close' to
    any trajectories in traj_base such that each trajectory in traj_base
    is used at most once.

    Used to evaluate the performance of algorithms.

    Arguments:
        traj_query : list
            A list of trajectories to compare.
        traj_base : list
            The second list of trajectories to compare.
        threshold : float
            The distance threshold between two observations
            to count a match (in pixels).
        times : list
            The list of zero-shifted times at which to evaluate the
            matches. The average of the distances at these times
            are used.

    Returns:
        list : a list of trajectories that appear in traj_query but not
               in traj_base where each trajectory in each set is only
               used once.
    """
    num_times = len(times)
    size_base = len(traj_base)
    used = [False] * size_base

    results = []
    for query in traj_query:
        best_dist = 10.0 * threshold
        best_ind = -1

        # Check the current query against all unused base trajectories.
        for j in range(size_base):
            if not used[j]:
                dist = ave_trajectory_distance(query, traj_base[j], times)
                if dist < best_dist:
                    best_dist = dist
                    best_ind = j

        # If we found a good match, save it.
        if best_dist <= threshold:
            used[best_ind] = True
        else:
            results.append(query)
    return results
