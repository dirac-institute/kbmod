import unittest

from kbmod.filters.clustering_filters import *
from kbmod.results import Results
from kbmod.search import *


class test_clustering_filters(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

    def _make_result_data(self, objs):
        """Create a ResultList for the given objects.

        Parameters
        ----------
        obj : `list[list]`
            A list where each element specifies a `Trajectory`
            as [x, y, xv, yv].

        Returns
        -------
        `Results`
        """
        trj_list = [Trajectory(x[0], x[1], x[2], x[3], lh=100.0) for x in objs]
        return Results.from_trajectories(trj_list)

    def test_dbscan_position_results(self):
        """Test clustering based on the initial position of the trajectory."""
        rs = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 1000, -1000],
                [10, 11, 0.0, 0.0],
                [25, 24, 1.0, 1.0],
                [25, 26, 10.0, 10.0],
                [10, 12, 5, 5],
            ]
        )

        # Standard-ish params collapses to 2 clusters.
        f1 = ClusterPredictionFilter(cluster_eps=5.0, pred_times=[0.0])
        self.assertEqual(f1.keep_indices(rs), [0, 3])

        # Small eps is 4 clusters.
        f2 = ClusterPredictionFilter(cluster_eps=0.000015, pred_times=[0.0])
        self.assertEqual(f2.keep_indices(rs), [0, 3, 4, 5])

        # Large scale means 1 cluster
        f3 = ClusterPredictionFilter(cluster_eps=5000.0, pred_times=[0.0])
        self.assertEqual(f3.keep_indices(rs), [0])

        # Catch invalid parameters
        self.assertRaises(ValueError, ClusterPredictionFilter, cluster_eps=0.025, pred_times=[])

    def test_dbscan_all_results(self):
        """Test clustering based on the median position of the trajectory."""
        rs = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 1000, -1000],
                [10, 11, 1.0, 2.1],
                [55, 54, 1.0, 1.0],
                [55, 56, 10.0, 10.0],
                [10, 12, 4.1, 8],
            ]
        )

        # Start with 5 clusters
        f1 = ClusterPosVelFilter(cluster_eps=5.0)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 4, 5])

        # Larger eps is 3 clusters.
        f2 = ClusterPosVelFilter(cluster_eps=20.0)
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3])

        # Adding the scaling factor increases or reduces the impact of velocity.
        f3 = ClusterPosVelFilter(cluster_eps=5.0, cluster_v_scale=5.0)
        self.assertEqual(f3.keep_indices(rs), [0, 1, 3, 4, 5])

        f4 = ClusterPosVelFilter(cluster_eps=5.0, cluster_v_scale=1e-9)
        self.assertEqual(f4.keep_indices(rs), [0, 3])

    def test_cluster_build_data(self):
        """Test that we predict the correct positions."""
        rs = self._make_result_data(
            [
                [10, 11, 0, 0],
                [10, 11, 1, 2],
            ]
        )
        times = [0.0, 0.5, 1.0, 1.5, 2.0]

        f1 = ClusterPosVelFilter(cluster_eps=5.0, cluster_v_scale=1.0)
        cluster_data = f1._build_clustering_data(rs)
        expected_data = np.array(
            [
                [10.0, 11.0, 0.0, 0.0],
                [10.0, 11.0, 1.0, 2.0],
            ]
        )
        self.assertEqual(cluster_data.dtype, np.float32)
        self.assertTrue(np.allclose(cluster_data, expected_data))

    def test_cluster_build_data_prediction(self):
        """Test that we predict the correct positions."""
        rs = self._make_result_data(
            [
                [10, 11, 0, 0],
                [10, 11, 1, 2],
            ]
        )
        times = [0.0, 0.5, 1.0, 1.5, 2.0]

        f1 = ClusterPredictionFilter(cluster_eps=2.0, pred_times=times)
        predicted_pos = f1._build_clustering_data(rs)
        expected_pos = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0],
                [10.0, 10.5, 11.0, 11.5, 12.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ]
        )
        self.assertEqual(predicted_pos.dtype, np.float32)
        self.assertTrue(np.allclose(predicted_pos, expected_pos))

    def test_dbscan_mid_pos(self):
        rs = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 2, 5],
                [10, 11, 1.01, 1.99],
                [21, 23, 1, 2],
                [21, 23, -10, -10],
                [5, 10, 6, 1],
                [5, 10, 1, 2],
            ]
        )

        # Use the median time.
        f1 = ClusterPredictionFilter(cluster_eps=2.0, pred_times=[0.95])
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 6])

        # Use different times.
        f2 = ClusterPredictionFilter(cluster_eps=2.0, pred_times=[10.0])
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3, 4, 5, 6])

        # Use different times again.
        f3 = ClusterPredictionFilter(cluster_eps=2.0, pred_times=[0.001])
        self.assertEqual(f3.keep_indices(rs), [0, 3, 5])

    def test_dbscan_start_end_pos(self):
        rs = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 2, 5],
                [10, 11, 1.01, 1.99],
                [10, 11, 0.99, 2.01],
                [21, 23, 1, 2],
                [21, 23, -10, -10],
                [21, 23, -10, -10.01],
                [21, 23, -10.01, -10],
                [5, 10, 1, 2.1],
                [5, 10, 1, 2],
                [5, 10, 1, 1.9],
            ]
        )

        f1 = ClusterPredictionFilter(cluster_eps=3.0, pred_times=[10, 11.9])
        self.assertEqual(f1.keep_indices(rs), [0, 1, 4, 5, 8])

    def test_clustering_results(self):
        cluster_params = {
            "cluster_type": "all",
            "cluster_eps": 5.0,
            "cluster_v_scale": 1.0,
            "times": self.times,
        }

        results = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 10, 20],
                [40, 5, -1, 2],
                [5, 0, 1, 2],
                [5, 1, 1, 2],
            ]
        )
        apply_clustering(results, cluster_params)
        self.assertEqual(len(results), 4)

        # If we remove the weighting on velocity, then we drop to three clusters.
        cluster_params["cluster_v_scale"] = 1e-16
        apply_clustering(results, cluster_params)
        self.assertEqual(len(results), 3)

        # Try clustering with only positions.
        results2 = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 10, 20],
                [40, 5, -1, 2],
                [5, 0, 1, 2],
                [5, 1, 1, 2],
            ]
        )
        cluster_params["cluster_type"] = "position"
        apply_clustering(results2, cluster_params)
        self.assertEqual(len(results2), 3)

        # Try invalid or missing cluster_type.
        cluster_params["cluster_type"] = "invalid"
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)

    def test_nnsweep_build_data(self):
        """Test that we predict the correct positions."""
        rs = self._make_result_data(
            [
                [10, 11, 0, 0],
                [10, 11, 1, 2],
            ]
        )
        times = [0.0, 0.5, 1.0, 1.5, 2.0]

        f1 = NNSweepFilter(cluster_eps=2.0, pred_times=times)
        predicted_pos = f1._build_clustering_data(rs)
        expected_pos = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0],
                [10.0, 10.5, 11.0, 11.5, 12.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ]
        )
        self.assertEqual(predicted_pos.dtype, np.float32)
        self.assertTrue(np.allclose(predicted_pos, expected_pos))

    def test_nnfilter(self):
        """Test filtering based on removing neighbors."""
        trjs = [
            Trajectory(x=10, y=11, vx=1.0, vy=2.0, lh=100.0),  # Cluster 0 - filtered
            Trajectory(x=10, y=12, vx=1.0, vy=2.0, lh=90.0),  # Cluster 0 - filtered
            Trajectory(x=11, y=11, vx=1.0, vy=2.0, lh=110.0),  # Cluster 0 - best
            Trajectory(x=10, y=11, vx=-10.0, vy=20.0, lh=90.0),  # Cluster 1 - best
            Trajectory(x=55, y=54, vx=10.0, vy=-2.0, lh=90.0),  # Cluster 2 - filtered
            Trajectory(x=55, y=56, vx=10.0, vy=-2.01, lh=95.0),  # Cluster 2 - best
            Trajectory(x=55, y=70, vx=10.0, vy=-2.01, lh=94.0),  # Cluster 3 - best
            Trajectory(x=155, y=58, vx=15.0, vy=-1.95, lh=50.0),  # Cluster 4 - best
            Trajectory(x=10, y=11, vx=-10.001, vy=20.0, lh=85.0),  # Cluster 1 - filtered
            Trajectory(x=155, y=58, vx=15.0, vy=-1.95, lh=10.0),  # Cluster 4 - filtered
            Trajectory(x=10, y=12, vx=1.01, vy=2.01, lh=90.0),  # Cluster 0 - filtered
            Trajectory(x=54, y=54, vx=9.99, vy=-2.0, lh=85.0),  # Cluster 2 - filtered
        ]
        rs = Results.from_trajectories(trjs)

        # Start with 5 clusters as noted above.
        f1 = NNSweepFilter(cluster_eps=5.0, pred_times=[0.0, 20.0])
        self.assertEqual(f1.keep_indices(rs), [2, 3, 5, 6, 7])

        # Larger eps includes more points.
        f2 = NNSweepFilter(cluster_eps=20.0, pred_times=[0.0, 20.0])
        self.assertEqual(f2.keep_indices(rs), [2, 3, 5, 7])

        # Using only the start time filters on the (x, y) values only.
        f3 = NNSweepFilter(cluster_eps=5.0, pred_times=[0.0])
        self.assertEqual(f3.keep_indices(rs), [2, 5, 6, 7])

        # Using five times.
        f4 = NNSweepFilter(cluster_eps=5.0, pred_times=[0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(f4.keep_indices(rs), [2, 3, 5, 6, 7])

        # Using five times and a tiny threshold. Everything should be its
        # own cluster except 9 which is an exact match with 7.
        f5 = NNSweepFilter(cluster_eps=1e-8, pred_times=[0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(f5.keep_indices(rs), [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11])

    def test_cluster_grid_filter(self):
        """Test filtering based on a discrete grid."""
        trjs = [
            Trajectory(x=10, y=11, vx=1.0, vy=2.0, lh=100.0),  # Cluster 0 - filtered
            Trajectory(x=10, y=12, vx=1.0, vy=2.0, lh=90.0),  # Cluster 0 - filtered
            Trajectory(x=11, y=11, vx=1.0, vy=2.0, lh=110.0),  # Cluster 0 - best
            Trajectory(x=10, y=12, vx=1.01, vy=2.01, lh=90.0),  # Cluster 0 - filtered
            Trajectory(x=55, y=57, vx=10.0, vy=-2.0, lh=90.0),  # Cluster 1 - filtered
            Trajectory(x=55, y=56, vx=10.0, vy=-2.01, lh=95.0),  # Cluster 1 - best
            # Close to cluster 0, but in a different bin.
            Trajectory(x=8, y=12, vx=1.0, vy=2.0, lh=90.0),  # Cluster 2 - best
            Trajectory(x=55, y=70, vx=10.0, vy=-2.01, lh=94.0),  # Cluster 3 - best
            Trajectory(x=155, y=59, vx=15.0, vy=-1.95, lh=10.0),  # Cluster 4 - filtered
            Trajectory(x=155, y=58, vx=15.0, vy=-1.95, lh=50.0),  # Cluster 4 - best
            Trajectory(x=156, y=58, vx=15.0, vy=-1.95, lh=10.0),  # Cluster 4 - filtered
            Trajectory(x=156, y=58, vx=-15.0, vy=-1.95, lh=10.0),  # Cluster 5 - best
        ]
        rs = Results.from_trajectories(trjs)

        # Start with 6 clusters as noted above.
        f1 = ClusterGridFilter(cluster_eps=5.0, pred_times=[0.0, 20.0])
        self.assertEqual(f1.keep_indices(rs), [2, 5, 6, 7, 9, 11])

        # Larger eps includes more points.
        f2 = ClusterGridFilter(cluster_eps=100.0, pred_times=[0.0, 20.0])
        self.assertEqual(f2.keep_indices(rs), [2, 5, 9, 11])

        # Using only the start time filters on the (x, y) values only.
        # We combine clusters 4 and 5.
        f3 = ClusterGridFilter(cluster_eps=5.0, pred_times=[0.0])
        self.assertEqual(f3.keep_indices(rs), [2, 5, 6, 7, 9])


if __name__ == "__main__":
    unittest.main()
