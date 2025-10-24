"""A data structure for online clustering of result trajectories."""

import numpy as np

from kbmod.search import (
    extract_all_trajectory_vx,
    extract_all_trajectory_vy,
    extract_all_trajectory_x,
    extract_all_trajectory_y,
)


class TrajectoryClusterGrid:
    """A spatial hash of trajectory results.

    Attributes
    ----------
    bin_width : `int`
        The width of each of the spatial bins (in pixels).
        Default is 10.
    max_time : `float`
        The maximum time difference between the first and last points in a trajectory.
        Default is 1.0.
    table : `dict`
        A dictionary mapping spatial bins to listthe best trajectory results
        that fell into that bin.
    count : `dict`
        A dictionary mapping spatial bins to the number of results that
        fell into that bin.
    total_count : `int`
        The total number of results that have been added to the table.
    """

    def __init__(self, bin_width=10, max_time=1.0):
        """Initialize the TrajectoryClusterTable.

        Parameters
        ----------
        bin_width : `int`, optional
            The width of each of the spatial bins (in pixels).
        """
        if bin_width < 1 or not np.isfinite(bin_width):
            raise ValueError(f"Bin width must be at least 1. Got {bin_width}.")
        self.bin_width = bin_width

        if max_time < 0 or not np.isfinite(max_time):
            raise ValueError(f"Max time must be >= 0. Got {max_time}.")
        self.max_time = max_time

        self.table = {}
        self.count = {}
        self.idx_table = {}
        self.total_count = 0

    def __len__(self):
        return len(self.table)

    def add_trajectory(self, trj, idx=None):
        """Add a result to the table.

        Parameters
        ----------
        trj : `Trajectory`
            The trajectory result to add.
        idx : `int`, optional
            The index of the result in the original list of results.
            Used for filtering.
        """
        if idx is None:
            idx = self.total_count

        # Compute the spatial bin.
        xs_bin = int(trj.x / self.bin_width)
        ys_bin = int(trj.y / self.bin_width)
        xe_bin = int((trj.x + self.max_time * trj.vx) / self.bin_width)
        ye_bin = int((trj.y + self.max_time * trj.vy) / self.bin_width)

        bin_key = (xs_bin, ys_bin, xe_bin, ye_bin)
        if bin_key not in self.table:
            # If this is the first time we have seen the bin, add the trajectory.
            self.table[bin_key] = trj
            self.count[bin_key] = 1
            self.idx_table[bin_key] = idx
        else:
            # Check if the new trajectory is better than the old one.
            old_result = self.table[bin_key]
            if trj.lh > old_result.lh:
                self.table[bin_key] = trj
                self.idx_table[bin_key] = idx
            self.count[bin_key] += 1

        self.total_count += 1

    def add_trajectory_list(self, trj_list):
        """Add a list of results to the table.

        Parameters
        ----------
        trj_list : `list` of `Trajectory`
            The trajectories to add.
        """
        # Extract the components for all results.
        x0 = np.asanyarray(extract_all_trajectory_x(trj_list))
        y0 = np.asanyarray(extract_all_trajectory_y(trj_list))
        vx = np.asanyarray(extract_all_trajectory_vx(trj_list))
        vy = np.asanyarray(extract_all_trajectory_vy(trj_list))

        # Compute the spatial bin (vectorized) for all results.
        xs_bin = (x0 / self.bin_width).astype(int)
        ys_bin = (y0 / self.bin_width).astype(int)
        xe_bin = ((x0 + self.max_time * vx) / self.bin_width).astype(int)
        ye_bin = ((y0 + self.max_time * vy) / self.bin_width).astype(int)

        # We need to insert the trajectories serially because we are modifying the table.
        for idx, trj in enumerate(trj_list):
            bin_key = (xs_bin[idx], ys_bin[idx], xe_bin[idx], ye_bin[idx])
            old_result = self.table.get(bin_key, None)

            if old_result is None:
                self.table[bin_key] = trj
                self.count[bin_key] = 1
                self.idx_table[bin_key] = idx
            elif trj.lh > old_result.lh:
                self.table[bin_key] = trj
                self.idx_table[bin_key] = idx
                self.count[bin_key] += 1
            else:
                self.count[bin_key] += 1

        self.total_count += len(trj_list)

    def get_trajectories(self):
        """Get all of the best trajectories from each bin.

        Returns
        -------
        `list`
            A list of the best trajectory results from each bin.
        """
        return list(self.table.values())

    def get_indices(self):
        """Get the indices of the best trajectories from each bin.

        Returns
        -------
        `list`
            A list of the indices of the best trajectory results from each bin.
        """
        return list(self.idx_table.values())


def apply_trajectory_grid_filter(trajectories, bin_width, max_dt):
    """Use the TrajectoryClusterGrid to remove near duplicates.

    Parameters
    ----------
    trajectories : `list` of `Trajectory`
        The trajectories to filter.
    bin_width : `int`
        The width of the bins in TrajectoryClusterGrid.
    max_dt : `float`
        The maximum time to use.

    Returns
    -------
    results : `list` of `Trajectory`
        The unfiltered trajectories.
    indices : `list` of `int`
        The indices of the unfiltered trajectories.
    """
    grid_filter = TrajectoryClusterGrid(bin_width=bin_width, max_time=max_dt)
    for idx, trj in enumerate(trajectories):
        grid_filter.add_trajectory(trj, idx=idx)
    return grid_filter.get_trajectories(), grid_filter.get_indices()
