import numpy as np
from sklearn.cluster import DBSCAN

from kbmod.results import Results
import kbmod.search as kb

logger = kb.Logging.getLogger(__name__)


class DBSCANFilter:
    """Cluster the candidates using DBSCAN and only keep a
    single representative trajectory from each cluster."""

    def __init__(self, eps, *args, **kwargs):
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
        return f"DBSCAN_{self.cluster_type}_{self.eps}"

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

        # Get the best index per cluster.
        top_vals = []
        for cluster_num in np.unique(cluster.labels_):
            cluster_vals = np.where(cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])
        return top_vals


class ClusterPositionFilter(DBSCANFilter):
    """Cluster the candidates using their scaled starting position"""

    def __init__(self, eps, height, width, *args, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold.
        height : `int`
            The size of the image height (in pixels) for scaling.
        width : `int`
            The size of the image width (in pixels) for scaling.
        """
        super().__init__(eps, *args, **kwargs)
        if height <= 0.0 or width <= 0:
            raise ValueError(f"Invalid scaling parameters y={height} by x={width}")
        self.height = height
        self.width = width
        self.cluster_type = "position"

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
        x_arr = np.array(result_data["x"]) / self.width
        y_arr = np.array(result_data["y"]) / self.height
        return np.array([x_arr, y_arr])


class ClusterPosAngVelFilter(DBSCANFilter):
    """Cluster the candidates using their scaled starting position, trajectory
    angles, and trajectory velocity magnitude.
    """

    def __init__(self, eps, height, width, vel_lims, ang_lims, *args, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold.
        height : `int`
            The size of the image height (in pixels) for scaling.
        width : `int`
            The size of the image width (in pixels) for scaling.
        vel_lims : `list`
            The velocity limits of the search such that v_lim[1] - v_lim[0]
            is the range of velocities searched. Used for scaling.
        ang_lims : `list`
            The angle limits of the search such that ang_lim[1] - ang_lim[0]
            is the range of velocities searched.
        """
        super().__init__(eps, *args, **kwargs)
        if height <= 0.0 or width <= 0:
            raise ValueError(f"Invalid scaling parameters y={height} by x={width}")
        if len(vel_lims) < 2:
            raise ValueError(f"Invalid velocity magnitude scaling parameters {vel_lims}")
        if len(ang_lims) < 2:
            raise ValueError(f"Invalid velocity angle scaling parameters {ang_lims}")
        self.height = height
        self.width = width

        self.v_scale = (vel_lims[1] - vel_lims[0]) if vel_lims[1] != vel_lims[0] else 1.0
        self.a_scale = (ang_lims[1] - ang_lims[0]) if ang_lims[1] != ang_lims[0] else 1.0
        self.min_v = vel_lims[0]
        self.min_a = ang_lims[0]

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
        vel_arr = np.sqrt(np.square(vx_arr) + np.square(vy_arr))
        ang_arr = np.arctan2(vy_arr, vx_arr)

        # Scale the values.
        scaled_x = x_arr / self.width
        scaled_y = y_arr / self.height
        scaled_vel = (vel_arr - self.min_v) / self.v_scale
        scaled_ang = (ang_arr - self.min_a) / self.a_scale

        return np.array([scaled_x, scaled_y, scaled_vel, scaled_ang])


class ClusterMidPosFilter(ClusterPositionFilter):
    """Cluster the candidates using their scaled positions at the median time."""

    def __init__(self, eps, height, width, times, *args, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold.
        height : `int`
            The size of the image height (in pixels) for scaling.
        width : `int`
            The size of the image width (in pixels) for scaling.
        times : `list` or `numpy.ndarray`
            A list of times for the images. Can be MJD or zero indexed.
        """
        super().__init__(eps, height, width, *args, **kwargs)

        zeroed_times = np.array(times) - times[0]
        self.midtime = np.median(zeroed_times)
        self.cluster_type = "midpoint"

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

        # Scale the values.
        scaled_mid_x = (x_arr + self.midtime * vx_arr) / self.width
        scaled_mid_y = (y_arr + self.midtime * vy_arr) / self.height

        return np.array([scaled_mid_x, scaled_mid_y])


def apply_clustering(result_data, cluster_params):
    """This function clusters results that have similar trajectories.

    Parameters
    ----------
    result_data: `Results`
        The set of results to filter. This data gets modified directly by
        the filtering.
    cluster_params : dict
        Contains values concerning the image and search settings including:
        cluster_type, eps, height, width, vel_lims, ang_lims, and mjd.

    Raises
    ------
    Raises a ValueError if the parameters are not valid.
    Raises a TypeError if ``result_data`` is of an unsupported type.
    """
    if "cluster_type" not in cluster_params:
        raise ValueError("Missing cluster_type parameter")
    cluster_type = cluster_params["cluster_type"]

    # Skip clustering if there is nothing to cluster.
    if len(result_data) == 0:
        logger.info("Clustering : skipping, no results.")
        return
    logger.info(f"Clustering {len(result_data)} results using {cluster_type}")

    # Do the clustering and the filtering.
    if cluster_type == "all":
        filt = ClusterPosAngVelFilter(**cluster_params)
    elif cluster_type == "position":
        filt = ClusterPositionFilter(**cluster_params)
    elif cluster_type == "mid_position":
        filt = ClusterMidPosFilter(**cluster_params)
    else:
        raise ValueError(f"Unknown clustering type: {cluster_type}")

    # Do the actual filtering.
    indices_to_keep = filt.keep_indices(result_data)
    result_data.filter_rows(indices_to_keep, filt.get_filter_name())
