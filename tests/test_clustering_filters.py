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
        f1 = ClusterPredictionFilter(eps=0.025, pred_times=[0.0], height=100, width=100)
        self.assertEqual(f1.keep_indices(rs), [0, 3])

        # Small eps is 4 clusters.
        f2 = ClusterPredictionFilter(eps=0.000015, pred_times=[0.0], height=100, width=100)
        self.assertEqual(f2.keep_indices(rs), [0, 3, 4, 5])

        # Large scale means 1 cluster
        f3 = ClusterPredictionFilter(eps=0.025, pred_times=[0.0], height=1000, width=1000)
        self.assertEqual(f3.keep_indices(rs), [0])

        # If we do not scale everything at the same starting point is marked its own cluster at low eps
        # (0.025 pixels max distance).
        f4 = ClusterPredictionFilter(eps=0.025, pred_times=[0.0], height=100, width=100, scaled=False)
        self.assertEqual(f4.keep_indices(rs), [0, 3, 4, 5])

        # We regain the clustering power with a higher eps (3 pixels max distance).
        f5 = ClusterPredictionFilter(eps=3.0, pred_times=[0.0], height=100, width=100, scaled=False)
        self.assertEqual(f5.keep_indices(rs), [0, 3])

        # Catch invalid parameters
        self.assertRaises(ValueError, ClusterPredictionFilter, eps=0.025, width=0)
        self.assertRaises(ValueError, ClusterPredictionFilter, eps=0.025, height=0)
        self.assertRaises(ValueError, ClusterPredictionFilter, eps=0.025, pred_times=[])

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
        f1 = ClusterPosAngVelFilter(eps=0.025, height=100, width=100, vel_lims=[0, 100], ang_lims=[0, 1.5])
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 4, 5])

        # Larger eps is 3 clusters.
        f2 = ClusterPosAngVelFilter(eps=0.25, height=100, width=100, vel_lims=[0, 100], ang_lims=[0, 1.5])
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3])

        # Larger scale is 3 clusters.
        f3 = ClusterPosAngVelFilter(eps=0.025, height=100, width=100, vel_lims=[0, 5000], ang_lims=[0, 1.5])
        self.assertEqual(f3.keep_indices(rs), [0, 1, 3])

        # Catch invalid parameters
        self.assertRaises(
            ValueError,
            ClusterPosAngVelFilter,
            eps=0.025,
            height=100,
            width=100,
            vel_lims=[100],
            ang_lims=[0, 1.5],
        )
        self.assertRaises(
            ValueError,
            ClusterPosAngVelFilter,
            eps=0.025,
            height=100,
            width=100,
            vel_lims=[0, 5000],
            ang_lims=[1.5],
        )

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
        f1 = ClusterPredictionFilter(eps=0.1, pred_times=[0.95], height=20, width=20)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 6])

        # Use different times.
        f2 = ClusterPredictionFilter(eps=0.1, pred_times=[10.0], height=20, width=20)
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3, 4, 5, 6])

        # Use different times again.
        f3 = ClusterPredictionFilter(eps=0.1, pred_times=[0.001], height=20, width=20)
        self.assertEqual(f3.keep_indices(rs), [0, 3, 5])

        # Try without scaling
        f4 = ClusterPredictionFilter(eps=0.1, pred_times=[10.95], height=20, width=20, scaled=False)
        self.assertEqual(f4.keep_indices(rs), [0, 1, 2, 3, 4, 5, 6])

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

        f1 = ClusterPredictionFilter(eps=3.0, pred_times=[10, 11.9], scaled=False)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 4, 5, 8])

    def test_clustering_results(self):
        cluster_params = {
            "ang_lims": [0.0, 1.5],
            "cluster_type": "all",
            "eps": 0.03,
            "times": self.times,
            "width": 100,
            "height": 100,
            "vel_lims": [5.0, 40.0],
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

        # Try clustering with start and end positions
        results3 = self._make_result_data(
            [
                [10, 11, 1, 2],
                [10, 11, 10, 20],
                [40, 5, -1, 2],
                [5, 0, 1, 2],
                [5, 1, 1, 2],
            ]
        )
        cluster_params["cluster_type"] = "start_end_position"
        apply_clustering(results3, cluster_params)
        self.assertEqual(len(results3), 4)

        # Try invalid or missing cluster_type.
        cluster_params["cluster_type"] = "invalid"
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)

        del cluster_params["cluster_type"]
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)


if __name__ == "__main__":
    unittest.main()
