import numpy as np
from sklearn.cluster import DBSCAN

from kbmod.results import Results
import kbmod.search as kb

logger = kb.Logging.getLogger(__name__)


class DBSCANFilter:
    """Cluster the candidates using DBSCAN and only keep a
    single representative trajectory from each cluster."""

    def __init__(self, eps, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold.
        """
        self.eps = eps
        self.cluster_type = ""
        self.cluster_args = dict(eps=self.eps, min_samples=1, n_jobs=-1)

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"DBSCAN_{self.cluster_type} eps={self.eps}"

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
    """Cluster the candidates using their positions at specific times."""

    def __init__(self, eps, pred_times=[0.0], **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold.
        pred_times : `list`
            The times a which to prediction the positions.
            Default = [0.0] (starting position only)
        """
        super().__init__(eps, **kwargs)

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
        x_arr = np.array(result_data["x"])
        y_arr = np.array(result_data["y"])
        vx_arr = np.array(result_data["vx"])
        vy_arr = np.array(result_data["vy"])

        # Append the predicted x and y location at each time. If scaling is turned off
        # the division by height and width will be no-ops.
        coords = []
        for t in self.times:
            coords.append(x_arr + t * vx_arr)
            coords.append(y_arr + t * vy_arr)
        return np.array(coords)


class ClusterPosVelFilter(DBSCANFilter):
    """Cluster the candidates using their starting position and velocities."""

    def __init__(self, eps, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold in pixels.
        """
        super().__init__(eps, **kwargs)
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
        x_arr = np.array(result_data["x"])
        y_arr = np.array(result_data["y"])
        vx_arr = np.array(result_data["vx"])
        vy_arr = np.array(result_data["vy"])
        return np.array([x_arr, y_arr, vx_arr, vy_arr])


def apply_clustering(result_data, cluster_params):
    """This function clusters results that have similar trajectories.

    Parameters
    ----------
    result_data: `Results`
        The set of results to filter. This data gets modified directly by
        the filtering.
    cluster_params : dict
        Contains values concerning the image and search settings including:
        cluster_type, eps, and times.

    Raises
    ------
    Raises a ValueError if the parameters are not valid.
    Raises a TypeError if ``result_data`` is of an unsupported type.
    """
    if "cluster_type" not in cluster_params:
        raise KeyError("Missing cluster_type parameter")
    cluster_type = cluster_params["cluster_type"]

    if "eps" not in cluster_params:
        raise KeyError("Missing eps parameter")
    eps = cluster_params["eps"]

    # Skip clustering if there is nothing to cluster.
    if len(result_data) == 0:
        logger.info("Clustering : skipping, no results.")
        return

    # Do the clustering and the filtering.
    if cluster_type == "all" or cluster_type == "pos_vel":
        filt = ClusterPosVelFilter(eps)
    elif cluster_type == "position" or cluster_type == "start_position":
        filt = ClusterPredictionFilter(eps, pred_times=[0.0])
    elif cluster_type == "mid_position":
        if not "times" in cluster_params:
            raise KeyError("Missing cluster_type parameter")
        all_times = np.sort(cluster_params["times"])
        zeroed_times = np.array(all_times) - all_times[0]
        median_time = np.median(zeroed_times)

        filt = ClusterPredictionFilter(eps, pred_times=[median_time])
    else:
        raise ValueError(f"Unknown clustering type: {cluster_type}")
    logger.info(f"Clustering {len(result_data)} results using {filt.get_filter_name()}")

    # Do the actual filtering.
    indices_to_keep = filt.keep_indices(result_data)
    result_data.filter_rows(indices_to_keep, filt.get_filter_name())
