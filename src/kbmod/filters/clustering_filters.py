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

    def __init__(self, eps, pred_times=[0.0], height=1.0, width=1.0, scaled=True, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        eps : `float`
            The clustering threshold.
        pred_times : `list`
            The times a which to prediction the positions.
            Default = [0.0] (starting position only)
        height : `int`
            The size of the image height (in pixels) for scaling.
            Default: 1 (no scaling)
        width : `int`
            The size of the image width (in pixels) for scaling.
            Default: 1 (no scaling)
        scaled : `bool`
            Scale the positions to [0, 1] based on ``width`` and ``height``. This impacts
            how ``eps`` is interpreted by DBSCAN. If scaling is turned on ``eps``
            approximates the percentage of each dimension between points. If scaling is
            turned off ``eps`` is a distance in pixels.
        """
        super().__init__(eps, **kwargs)
        if scaled:
            if height <= 0.0 or width <= 0:
                raise ValueError(f"Invalid scaling parameters y={height} by x={width}")
            self.height = height
            self.width = width
        else:
            self.height = 1.0
            self.width = 1.0

        # Confirm we have at least one prediction time.
        if len(pred_times) == 0:
            raise ValueError("No prediction times given.")
        self.times = pred_times

        # Set up the clustering algorithm's name.
        self.cluster_type = f"position (scaled={scaled}) t={self.times}"

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
            coords.append((x_arr + t * vx_arr) / self.width)
            coords.append((y_arr + t * vy_arr) / self.height)
        return np.array(coords)


class ClusterPosAngVelFilter(DBSCANFilter):
    """Cluster the candidates using their scaled starting position, trajectory
    angles, and trajectory velocity magnitude.
    """

    def __init__(self, eps, height, width, vel_lims, ang_lims, **kwargs):
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
        super().__init__(eps, **kwargs)
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

        # Scale the values. We always do this because distances and velocities
        # are not directly comparable.
        scaled_x = x_arr / self.width
        scaled_y = y_arr / self.height
        scaled_vel = (vel_arr - self.min_v) / self.v_scale
        scaled_ang = (ang_arr - self.min_a) / self.a_scale

        return np.array([scaled_x, scaled_y, scaled_vel, scaled_ang])


def apply_clustering(result_data, cluster_params):
    """This function clusters results that have similar trajectories.

    Parameters
    ----------
    result_data: `Results`
        The set of results to filter. This data gets modified directly by
        the filtering.
    cluster_params : dict
        Contains values concerning the image and search settings including:
        cluster_type, eps, height, width, scaled, vel_lims, ang_lims, and times.

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

    # Get the times used for prediction clustering.
    all_times = np.sort(cluster_params["times"])
    zeroed_times = np.array(all_times) - all_times[0]

    # Do the clustering and the filtering.
    if cluster_type == "all" or cluster_type == "pos_vel":
        filt = ClusterPosAngVelFilter(**cluster_params)
    elif cluster_type == "position" or cluster_type == "start_position":
        cluster_params["pred_times"] = [0.0]
        cluster_params["scaled"] = True
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "position_unscaled" or cluster_type == "start_position_unscaled":
        cluster_params["pred_times"] = [0.0]
        cluster_params["scaled"] = False
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "mid_position":
        cluster_params["pred_times"] = [np.median(zeroed_times)]
        cluster_params["scaled"] = True
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "mid_position_unscaled":
        cluster_params["pred_times"] = [np.median(zeroed_times)]
        cluster_params["scaled"] = False
        filt = ClusterPredictionFilter(**cluster_params)
    elif cluster_type == "start_end_position":
        cluster_params["pred_times"] = [0.0, zeroed_times[-1]]
        filt = ClusterPredictionFilter(**cluster_params)
        cluster_params["scaled"] = True
    elif cluster_type == "start_end_position_unscaled":
        cluster_params["pred_times"] = [0.0, zeroed_times[-1]]
        filt = ClusterPredictionFilter(**cluster_params)
        cluster_params["scaled"] = False
    else:
        raise ValueError(f"Unknown clustering type: {cluster_type}")
    logger.info(f"Clustering {len(result_data)} results using {filt.get_filter_name()}")

    # Do the actual filtering.
    indices_to_keep = filt.keep_indices(result_data)
    result_data.filter_rows(indices_to_keep, filt.get_filter_name())
