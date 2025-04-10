import numpy as np

from sklearn.cluster import DBSCAN

from kbmod.filters.clustering_grid import TrajectoryClusterGrid
from kbmod.results import Results
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
        data = self._build_clustering_data(result_data)

        # Set up the clustering algorithm
        cluster = DBSCAN(**self.cluster_args)
        cluster.fit(np.array(data, dtype=float).T)

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
        self.times = pred_times

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
        x_arr = np.asarray(result_data["x"])
        y_arr = np.asarray(result_data["y"])
        vx_arr = np.asarray(result_data["vx"])
        vy_arr = np.asarray(result_data["vy"])

        # Append the predicted x and y location at each time.
        coords = []
        for t in self.times:
            coords.append(x_arr + t * vx_arr)
            coords.append(y_arr + t * vy_arr)
        return np.array(coords)


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
        # Create arrays of each the trajectories information.
        x_arr = np.asarray(result_data["x"])
        y_arr = np.asarray(result_data["y"])
        vx_arr = np.asarray(result_data["vx"]) * self.cluster_v_scale
        vy_arr = np.asarray(result_data["vy"]) * self.cluster_v_scale
        return np.array([x_arr, y_arr, vx_arr, vy_arr])


class NNSweepFilter:
    """Filter any points that have neighboring trajectory with
    a higher likleihood within the threshold.

    Parameters
    ----------
    thresh : `float`
        The filtering threshold to use (in pixels).
    times : list-like
        The times at which to evaluate the trajectories (in days).
    """

    def __init__(self, cluster_eps, pred_times):
        """Create a NNFilter.

        Parameters
        ----------
        cluster_eps : `float`
            The filtering threshold to use.
        pred_times : list-like
            The times at which to evaluate the trajectories.
        """
        if cluster_eps <= 0.0:
            raise ValueError(f"Threshold must be > 0.0.")
        self.thresh = cluster_eps

        self.times = np.asarray(pred_times)
        if len(self.times) == 0:
            raise ValueError(f"Empty time array provided.")

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
        x_arr = np.asarray(result_data["x"])
        y_arr = np.asarray(result_data["y"])
        vx_arr = np.asarray(result_data["vx"])
        vy_arr = np.asarray(result_data["vy"])

        # Append the predicted x and y location at each time.
        coords = np.empty((len(result_data), 2 * len(self.times)))
        for t_idx, t_val in enumerate(self.times):
            coords[:, 2 * t_idx] = x_arr + t_val * vx_arr
            coords[:, 2 * t_idx + 1] = y_arr + t_val * vy_arr
        return coords

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
        cart_data = self._build_clustering_data(result_data)
        kd_tree = KDTree(cart_data)

        num_pts = len(result_data)
        lh_data = np.asarray(result_data["likelihood"])

        # For each point, search for all neighbors within the threshold and
        # only keep the point if it has the highest likelihood in that range.
        can_skip = np.full(num_pts, False)
        keep_vals = []
        for idx in range(num_pts):
            if not can_skip[idx]:
                # Run a range search to find all nearby neighbors.
                matches = kd_tree.query_ball_point(cart_data[idx, :], self.thresh)
                best_match = matches[np.argmax(lh_data[matches])]
                if best_match == idx:
                    keep_vals.append(idx)

                    # Everything found in this run doesn't need to be searched,
                    # because we have found the maximum value in this area.
                    can_skip[matches] = True
        return keep_vals


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
