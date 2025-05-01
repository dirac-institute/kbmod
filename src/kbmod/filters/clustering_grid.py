"""A data structure for online clustering of result trajectories."""


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
        if bin_width < 1:
            raise ValueError("Bin width must be at least 1.")
        self.bin_width = bin_width

        if max_time < 0:
            raise ValueError("Max time must be positive.")
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
