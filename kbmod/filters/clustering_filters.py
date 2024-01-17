import numpy as np
from sklearn.cluster import DBSCAN

from kbmod.filters.base_filter import BatchFilter
from kbmod.result_list import ResultList, ResultRow


class DBSCANFilter(BatchFilter):
    """Cluster the candidates using DBSCAN and only keep a
    single representative trajectory from each cluster."""

    def __init__(self, cluster_type, eps, x_size, y_size, vel_lims, ang_lims, mjd_times, *args, **kwargs):
        """Create a DBSCANFilter.

        Parameters
        ----------
        cluster_type : ``str``
            A string indicating the type of clustering to perform: all, position, or
            mid_position.
        eps : ``float``
            The clustering threshold.
        x_size : ``int``
            The width of the images (in pixels) used in the kbmod stack. Used
            for scaling.
        y_size : ``int``
            The height of the images (in pixels) used in the kbmod stack. Used
            for scaling.
        vel_lims : list
            The velocity limits of the search such that v_lim[1] - v_lim[0]
            is the range of velocities searched. Used for scaling.
        ang_lims : list
            The angle limits of the search such that ang_lim[1] - ang_lim[0]
            is the range of velocities searched.
        mjd_times : list
            A list of MJD times for the images.
        """
        super().__init__(*args, **kwargs)

        self.cluster_type = cluster_type
        self.eps = eps
        self.x_size = x_size
        self.y_size = y_size
        self.zeroed_times = np.array(mjd_times) - mjd_times[0]
        self.vel_lims = vel_lims
        self.ang_lims = ang_lims

    def get_filter_name(self):
        """Get the name of the filter.

        Returns
        -------
        str
            The filter name.
        """
        return f"DBSCAN_{self.cluster_type}_{self.eps}"

    def keep_indices(self, result_list: ResultList):
        """Determine which of the ResultList's indices to keep.

        Parameters
        ----------
        result_list: ResultList
            The set of results to filter.

        Returns
        -------
        list
           A list of indices (int) indicating which rows to keep.
        """
        cluster_args = dict(eps=self.eps, min_samples=1, n_jobs=-1)

        # Create arrays of each the trajectories information.
        x_arr = np.array([row.trajectory.x for row in result_list.results])
        y_arr = np.array([row.trajectory.y for row in result_list.results])
        vx_arr = np.array([row.trajectory.vx for row in result_list.results])
        vy_arr = np.array([row.trajectory.vy for row in result_list.results])
        vel_arr = np.sqrt(np.square(vx_arr) + np.square(vy_arr))
        ang_arr = np.arctan2(vy_arr, vx_arr)

        # Scale the values.
        scaled_x = x_arr / self.x_size
        scaled_y = y_arr / self.y_size

        v_scale = (self.vel_lims[1] - self.vel_lims[0]) if self.vel_lims[1] != self.vel_lims[0] else 1.0
        a_scale = (self.ang_lims[1] - self.ang_lims[0]) if self.ang_lims[1] != self.ang_lims[0] else 1.0
        scaled_vel = (vel_arr - self.vel_lims[0]) / v_scale
        scaled_ang = (ang_arr - self.ang_lims[0]) / a_scale

        # Do the clustering.
        cluster = DBSCAN(**cluster_args)
        if self.cluster_type == "all":
            cluster.fit(np.array([scaled_x, scaled_y, scaled_vel, scaled_ang], dtype=float).T)
        elif self.cluster_type == "position":
            cluster.fit(np.array([scaled_x, scaled_y], dtype=float).T)
        elif self.cluster_type == "mid_position":
            median_time = np.median(self.zeroed_times)
            scaled_mid_x = (x_arr + median_time * vx_arr) / self.x_size
            scaled_mid_y = (y_arr + median_time * vy_arr) / self.y_size
            cluster.fit(np.array([scaled_mid_x, scaled_mid_y], dtype=float).T)

        # Get the best index per cluster.
        top_vals = []
        for cluster_num in np.unique(cluster.labels_):
            cluster_vals = np.where(cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])
        return top_vals
