import numpy as np

from sklearn.cluster import DBSCAN

from kbmod.filters.clustering_grid import TrajectoryClusterGrid
import kbmod.search as kb

logger = kb.Logging.getLogger(__name__)


class DBSCANFilter:
    """Cluster the candidates using DBSCAN and only keep a
    single representative trajectory from each cluster.

    Attributes
    ----------
    cluster_eps : `float`
        The clustering threshold (in pixels).
    cluster_type : `str`
        The type of clustering.
    cluster_args : `dict`
        Additional arguments to pass to the clustering algorithm.
    """

    def __init__(self, cluster_eps, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        cluster_eps : `float`
            The clustering threshold.
        **kwargs : `dict`
            Additional arguments to pass to the clustering algorithm.
        """
        self.cluster_eps = cluster_eps
        self.cluster_type = ""
        self.cluster_args = dict(eps=self.cluster_eps, min_samples=1, n_jobs=-1)

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"DBSCAN_{self.cluster_type} eps={self.cluster_eps}"

    def _build_clustering_data(self, result_data):
        """Build the specific data set for this clustering approach.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        data : `numpy.ndarray`
           The N x D matrix to cluster where N is the number of results
           and D is the number of attributes.
        """
        raise NotImplementedError()

    def keep_indices(self, result_data):
        """Determine which of the results's indices to keep.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        `list`
           A list of indices (int) indicating which rows to keep.
        """
        # Build a numpy array of the trajectories to cluster with one row for each trajectory.
        data = self._build_clustering_data(result_data)

        # Set up the clustering algorithm
        cluster = DBSCAN(**self.cluster_args)
        cluster.fit(data)

        # Get the best index per cluster. If the data is sorted by LH, this should always
        # be the first point in the cluster. But we do an argmax in case the user has
        # manually sorted the data by something else.
        top_vals = []
        for cluster_num in np.unique(cluster.labels_):
            cluster_vals = np.where(cluster.labels_ == cluster_num)[0]
            top_ind = np.argmax(result_data["likelihood"][cluster_vals])
            top_vals.append(cluster_vals[top_ind])
        return top_vals


class ClusterPredictionFilter(DBSCANFilter):
    """Cluster the candidates using their positions at specific times.

    Attributes
    ----------
    times : list-like
        The times at which to evaluate the trajectories (in days).
    """

    def __init__(self, cluster_eps, pred_times=[0.0], **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        cluster_eps : `float`
            The clustering threshold.
        pred_times : `list`
            The times a which to prediction the positions (in days).
            Default = [0.0] (starting position only)
        """
        super().__init__(cluster_eps, **kwargs)

        # Confirm we have at least one prediction time.
        if len(pred_times) == 0:
            raise ValueError("No prediction times given.")
        self.times = np.array(pred_times, dtype=np.float32)

        # Set up the clustering algorithm's name.
        self.cluster_type = f"position t={self.times}"

    def _build_clustering_data(self, result_data):
        """Build the specific data set for this clustering approach.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        data : `numpy.ndarray`
           The N x D matrix to cluster where N is the number of results
           and D is the number of attributes.
        """
        x0_arr = result_data["x"][:, np.newaxis].astype(np.float32)
        xv_arr = result_data["vx"][:, np.newaxis].astype(np.float32)
        pred_x = x0_arr + xv_arr * self.times[np.newaxis, :]

        y0_arr = result_data["y"][:, np.newaxis].astype(np.float32)
        yv_arr = result_data["vy"][:, np.newaxis].astype(np.float32)
        pred_y = y0_arr + yv_arr * self.times[np.newaxis, :]
        return np.hstack([pred_x, pred_y])


class ClusterPosVelFilter(DBSCANFilter):
    """Cluster the candidates using their starting position and velocities."""

    def __init__(self, cluster_eps, cluster_v_scale=1.0, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        cluster_eps : `float`
            The clustering threshold (in pixels).
        cluster_v_scale : `float`
            The relative scaling of velocity differences compared to position
            differences. Default: 1.0 (no difference).
        """
        super().__init__(cluster_eps, **kwargs)
        if cluster_v_scale < 0.0:
            raise ValueError("cluster_v_scale cannot be negative.")
        self.cluster_v_scale = cluster_v_scale
        self.cluster_type = "all"

    def _build_clustering_data(self, result_data):
        """Build the specific data set for this clustering approach.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        data : `numpy.ndarray`
           The N x D matrix to cluster where N is the number of results
           and D is the number of attributes.
        """
        data = np.empty((len(result_data), 4), dtype=np.float32)
        data[:, 0] = result_data["x"].astype(np.float32)
        data[:, 1] = result_data["y"].astype(np.float32)
        data[:, 2] = result_data["vx"] * self.cluster_v_scale
        data[:, 3] = result_data["vy"] * self.cluster_v_scale
        return data


class NNSweepFilter:
    """Filter any points that have neighboring trajectory with
    a higher likleihood within the threshold.

    Attributes
    ----------
    thresh : `float`
        The filtering threshold to use (in pixels).
    times : list-like
        The times at which to evaluate the trajectories (in days).
    batch_size : `int`
        The size of batching to use for kd-tree lookups.  A batch size of 1
        turns off multi-threading and runs everything in series.
        Default: 1000
    """

    def __init__(self, cluster_eps, pred_times, batch_size=1_000):
        """Create a NNFilter.

        Parameters
        ----------
        cluster_eps : `float`
            The filtering threshold to use.
        pred_times : list-like
            The times at which to evaluate the trajectories.
        batch_size : `int`
            The size of batching to use for kd-tree lookups.  A batch size of 1
            turns off multi-threading and runs everything in series.
            Default: 1000
        """
        if cluster_eps <= 0.0:
            raise ValueError(f"Threshold must be > 0.0.")
        self.thresh = cluster_eps

        self.times = np.asarray(pred_times, dtype=np.float32)
        if len(self.times) == 0:
            raise ValueError(f"Empty time array provided.")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0.")
        self.batch_size = batch_size

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"NNFilter times={self.times} eps={self.thresh}"

    def _build_clustering_data(self, result_data):
        """Build the specific data set for this clustering approach.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        data : `numpy.ndarray`
           The N x D matrix to cluster where N is the number of results
           and D is the number of attributes.
        """
        x0_arr = result_data["x"][:, np.newaxis].astype(np.float32)
        xv_arr = result_data["vx"][:, np.newaxis].astype(np.float32)
        pred_x = x0_arr + xv_arr * self.times[np.newaxis, :]

        y0_arr = result_data["y"][:, np.newaxis].astype(np.float32)
        yv_arr = result_data["vy"][:, np.newaxis].astype(np.float32)
        pred_y = y0_arr + yv_arr * self.times[np.newaxis, :]
        return np.hstack([pred_x, pred_y])

    def keep_indices(self, result_data):
        """Determine which of the results's indices to keep.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        `list`
           A list of indices (int) indicating which rows to keep.
        """
        from scipy.spatial import KDTree

        # Predict the Trajectory's locations at the given times and put the
        # resulting points in a KDTree.
        build_data_timer = kb.DebugTimer("NNSweepFilter building data", logger)
        cart_data = self._build_clustering_data(result_data)
        kd_tree = KDTree(cart_data)
        build_data_timer.stop()

        num_pts = len(result_data)
        lh_data = result_data["likelihood"]

        # For each point, search for all neighbors within the threshold and
        # only keep the point if it has the highest likelihood in that range.
        # We do this in batches to benefit from multi-threaded KDTree queries
        # while avoiding too much memory for the match data.
        num_workers = -1 if self.batch_size > 1 else 1
        can_skip = np.full(num_pts, False)
        keep_vals = []
        batch_start = 0

        while batch_start < num_pts:
            # Get the next batch of indices to search.  Each batch only includes those
            # results that have not already been eliminated to avoid unnecessary searches.
            batch_end = min(num_pts, batch_start + self.batch_size)
            batch_inds = np.asanyarray([i for i in range(batch_start, batch_end) if not (can_skip[i])])

            # Skip all the work if there is nothing to query in this batch.
            if len(batch_inds) == 0:
                batch_start = batch_end
                continue

            # Do the (multi-threaded) KD-tree search for his batch of indices.
            batch_matches = kd_tree.query_ball_point(
                cart_data[batch_inds, :],
                self.thresh,
                workers=num_workers,
            )

            # Check if each index is the best in its neighborhood.
            for batch_idx, total_idx in enumerate(batch_inds):
                if not can_skip[total_idx]:
                    matches = np.asanyarray(batch_matches[batch_idx])
                    if lh_data[total_idx] >= np.max(lh_data[matches]):
                        keep_vals.append(total_idx)

                        # Everything found in this run (including the current point)
                        # doesn't need to be searched in the future, because we have
                        # found the maximum value in this area.
                        can_skip[matches] = True

            batch_start = batch_end

        return keep_vals


class GreedyValidEndpointFilter:
    """Experimental likelihood-ordered suppression on shared valid epochs.

    Each pair is compared at the earliest and latest times where both rows
    have ``obs_valid=True``. The distance is the Euclidean norm of the four
    endpoint coordinate differences, as in ``nn_start_end``. A pair must share
    at least ``min_shared_obs`` observations at two distinct times, and at least
    ``min_shared_fraction`` of *each* row's valid observations.

    Rows are visited in descending likelihood order (input order breaks ties).
    Only retained rows suppress neighbors, avoiding transitive chains through
    rejected candidates. This pair-dependent distance is not a KD-tree metric.
    The reference implementation costs O(N^2 T) in the worst case and batches
    comparisons to avoid allocating an N x N distance matrix.

    Parameters
    ----------
    cluster_eps : `float`
        Positive, finite endpoint distance threshold in pixels.
    pred_times : array-like
        One finite time offset per obs_valid column, in days relative to the
        trajectory's reference epoch. Column order is preserved.
    min_shared_obs : `int`, optional
        Minimum number of shared valid observations. Default is 2.
    min_shared_fraction : `float`, optional
        Minimum shared fraction of each row's valid observations. Default is 0.5.
    batch_size : `int`, optional
        Maximum number of pair comparisons per batch. Default is 1024.
    """

    def __init__(
        self, cluster_eps, pred_times, *, min_shared_obs=2, min_shared_fraction=0.5, batch_size=1024
    ):
        if not np.isfinite(cluster_eps) or cluster_eps <= 0:
            raise ValueError("cluster_eps must be finite and > 0.")
        self.thresh = cluster_eps
        self.times = np.asarray(pred_times, dtype=np.float64)
        if self.times.ndim != 1 or self.times.size == 0 or not np.all(np.isfinite(self.times)):
            raise ValueError("pred_times must be a nonempty, finite one-dimensional array.")
        if (
            isinstance(min_shared_obs, (bool, np.bool_))
            or not isinstance(min_shared_obs, (int, np.integer))
            or min_shared_obs < 2
        ):
            raise ValueError("min_shared_obs must be an integer >= 2.")
        if not np.isfinite(min_shared_fraction) or not 0 <= min_shared_fraction <= 1:
            raise ValueError("min_shared_fraction must be finite and in [0, 1].")
        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        self.min_shared_obs = min_shared_obs
        self.min_shared_fraction = min_shared_fraction
        self.batch_size = batch_size

    def get_filter_name(self):
        return (
            f"GreedyValidEndpointFilter eps={self.thresh} "
            f"min_shared_obs={self.min_shared_obs} min_shared_fraction={self.min_shared_fraction}"
        )

    def keep_indices(self, result_data):
        """Return retained row indices in input order without changing the data.

        Requires a boolean ``obs_valid`` matrix aligned with ``pred_times`` and
        finite trajectory coordinates and likelihoods. Rows with insufficient
        shared evidence are retained, including rows with no valid observations.
        """
        if len(result_data) == 0:
            return []
        if "obs_valid" not in result_data.colnames:
            raise ValueError("greedy_valid_start_end requires an obs_valid column.")
        valid = np.asarray(result_data["obs_valid"])
        if valid.dtype.kind != "b" or valid.shape != (len(result_data), len(self.times)):
            raise ValueError("obs_valid must be a boolean matrix with one column per prediction time.")
        coords = np.column_stack([result_data[c] for c in ("x", "y", "vx", "vy")]).astype(float)
        likelihood = np.asarray(result_data["likelihood"], dtype=float)
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(likelihood)):
            raise ValueError("Trajectory coordinates and likelihoods must be finite.")

        counts = valid.sum(axis=1)
        order = np.argsort(-likelihood, kind="stable")
        suppressed = np.zeros(len(result_data), dtype=bool)
        keep = []
        for rank, idx in enumerate(order):
            if suppressed[idx]:
                continue
            keep.append(int(idx))
            for start in range(rank + 1, len(order), self.batch_size):
                others = order[start : start + self.batch_size]
                others = others[~suppressed[others]]
                if len(others) == 0:
                    continue
                shared = valid[others] & valid[idx]
                num_shared = shared.sum(axis=1)
                enough = (num_shared >= self.min_shared_obs) & (
                    num_shared >= self.min_shared_fraction * np.maximum(counts[idx], counts[others])
                )
                others, shared = others[enough], shared[enough]
                if len(others) == 0:
                    continue
                first = np.min(np.where(shared, self.times, np.inf), axis=1)
                last = np.max(np.where(shared, self.times, -np.inf), axis=1)
                delta = coords[others] - coords[idx]
                dx_first = delta[:, 0] + delta[:, 2] * first
                dy_first = delta[:, 1] + delta[:, 3] * first
                dx_last = delta[:, 0] + delta[:, 2] * last
                dy_last = delta[:, 1] + delta[:, 3] * last
                distance = np.hypot(np.hypot(dx_first, dy_first), np.hypot(dx_last, dy_last))
                suppressed[others[(first < last) & (distance <= self.thresh)]] = True
        return sorted(keep)


class ClusterGridFilter:
    """Use a discrete grid to cluster the points. Each trajectory
    is fit into a bin and only the best trajectory per bin is retained.

    Attributes
    ----------
    bin_width : `int`
        The width of the grid bins (in pixels).
    cluster_grid : `TrajectoryClusterGrid`
        The grid of best result trajectories seen.
    max_dt : `float`
        The maximum different between times in pred_times.
    """

    def __init__(self, cluster_eps, pred_times):
        """Create a ClusterGridFilter.

        Parameters
        ----------
        cluster_eps : `float`
            The bin width to use (in pixels).
        pred_times : list-like
            The times at which to evaluate the trajectories (in days).
        """
        self.bin_width = np.ceil(cluster_eps)
        if self.bin_width <= 0:
            raise ValueError(f"Bin width must be > 0.0.")

        self.times = np.asarray(pred_times)
        if len(self.times) == 0:
            self.times = np.array([0.0])
        self.max_dt = np.max(self.times) - np.min(self.times)

        # Create the actual grid to store the results.
        self.cluster_grid = TrajectoryClusterGrid(
            bin_width=self.bin_width,
            max_time=self.max_dt,
        )

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"ClusterGridFilter bin_width{self.bin_width}, max_dt={self.max_dt}"

    def keep_indices(self, result_data):
        """Determine which of the results's indices to keep.

        Parameters
        ----------
        result_data: `Results`
            The set of results to filter.

        Returns
        -------
        `list`
           A list of indices (int) indicating which rows to keep.
        """
        trj_list = result_data.make_trajectory_list()
        for idx, trj in enumerate(trj_list):
            self.cluster_grid.add_trajectory(trj, idx)

        keep_vals = np.sort(self.cluster_grid.get_indices())
        return list(keep_vals)


def apply_clustering(result_data, cluster_params):
    """This function clusters results that have similar trajectories.

    Parameters
    ----------
    result_data: `Results`
        The set of results to filter. This data gets modified directly by
        the filtering.
    cluster_params : dict
        Contains values concerning the image and search settings including:
        cluster_type, cluster_eps, times, and cluster_v_scale (optional).
        The experimental greedy_valid_start_end mode also accepts
        cluster_min_shared_obs (default 2) and cluster_min_shared_fraction
        (default 0.5). Times must align with the obs_valid columns and x/y
        must refer to the first supplied time (the ImageStack reference epoch).

    Raises
    ------
    Raises a ValueError if the parameters are not valid.
    Raises a TypeError if ``result_data`` is of an unsupported type.
    """
    if "cluster_type" not in cluster_params:
        raise KeyError("Missing cluster_type parameter")
    cluster_type = cluster_params["cluster_type"]

    # Skip clustering if there is nothing to cluster.
    if len(result_data) == 0:
        logger.info("Clustering : skipping, no results.")
        return

    # Get the times used for prediction clustering.
    if not "times" in cluster_params:
        raise KeyError("Missing times parameter in the clustering parameters.")
    if cluster_type == "greedy_valid_start_end":
        # Do not sort times independently of the per-observation validity mask.
        times = np.asarray(cluster_params["times"], dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("times must be a nonempty, finite one-dimensional array.")
        filt = GreedyValidEndpointFilter(
            cluster_params["cluster_eps"],
            times - times[0],
            min_shared_obs=cluster_params.get("cluster_min_shared_obs", 2),
            min_shared_fraction=cluster_params.get("cluster_min_shared_fraction", 0.5),
        )
        logger.info(f"Clustering {len(result_data)} results using {filt.get_filter_name()}")
        result_data.filter_rows(filt.keep_indices(result_data), filt.get_filter_name())
        return
    all_times = np.sort(cluster_params["times"])
    zeroed_times = np.array(all_times) - all_times[0]

    # Do the clustering and the filtering.
    if cluster_type == "all" or cluster_type == "pos_vel":
        filt = ClusterPosVelFilter(**cluster_params)
    elif cluster_type == "position" or cluster_type == "start_position":
        cluster_params["pred_times"] = [0.0]
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "mid_position":
        cluster_params["pred_times"] = [np.median(zeroed_times)]
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "start_end_position":
        cluster_params["pred_times"] = [0.0, zeroed_times[-1]]
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "nn_start_end":
        filt = NNSweepFilter(cluster_params["cluster_eps"], [0.0, zeroed_times[-1]])
    elif cluster_type == "nn_start":
        filt = NNSweepFilter(cluster_params["cluster_eps"], [0.0])
    elif cluster_type == "grid_start_end":
        filt = ClusterGridFilter(cluster_params["cluster_eps"], [0.0, zeroed_times[-1]])
    elif cluster_type == "grid_start":
        filt = ClusterGridFilter(cluster_params["cluster_eps"], [0.0])
    else:
        raise ValueError(f"Unknown clustering type: {cluster_type}")
    logger.info(f"Clustering {len(result_data)} results using {filt.get_filter_name()}")

    # Do the actual filtering.
    indices_to_keep = filt.keep_indices(result_data)
    result_data.filter_rows(indices_to_keep, filt.get_filter_name())
