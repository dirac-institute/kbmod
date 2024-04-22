import unittest

from kbmod.filters.clustering_filters import *
from kbmod.result_list import ResultList, ResultRow
from kbmod.results import Results
from kbmod.search import *
from kbmod.trajectory_utils import make_trajectory


class test_clustering_filters(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

    def _make_result_list(self, objs):
        """Create a ResultList for the given objects.

        Parameters
        ----------
        obj : `list[list]`
            A list where each element specifies a `Trajectory`
            as [x, y, xv, yv].

        Returns
        -------
        `ResultList`
        """
        rs = ResultList(self.times, track_filtered=True)
        for x in objs:
            trj = make_trajectory(x[0], x[1], x[2], x[3], lh=100.0)
            row = ResultRow(trj, self.num_times)
            rs.append_result(row)
        return rs

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
        trj_list = [make_trajectory(x[0], x[1], x[2], x[3], lh=100.0) for x in objs]
        return Results.from_trajectories(trj_list)

    def test_dbscan_position_result_list(self):
        rs = self._make_result_list(
            [
                [10, 11, 1, 2],
                [10, 11, 1000, -1000],
                [10, 11, 0.0, 0.0],
                [25, 24, 1.0, 1.0],
                [25, 26, 10.0, 10.0],
                [10, 12, 5, 5],
            ]
        )
        self.assertTrue(type(rs) is ResultList)

        # Standard-ish params collapses to 2 clusters.
        f1 = ClusterPositionFilter(eps=0.025, height=100, width=100)
        self.assertEqual(f1.keep_indices(rs), [0, 3])

        # Small eps is 4 clusters.
        f2 = ClusterPositionFilter(eps=0.000015, height=100, width=100)
        self.assertEqual(f2.keep_indices(rs), [0, 3, 4, 5])

        # Large scale means 1 cluster
        f3 = ClusterPositionFilter(eps=0.025, height=1000, width=1000)
        self.assertEqual(f3.keep_indices(rs), [0])

        # Catch invalid parameters
        self.assertRaises(ValueError, ClusterPositionFilter, eps=0.025, height=100, width=0)
        self.assertRaises(ValueError, ClusterPositionFilter, eps=0.025, height=0, width=100)

    def test_dbscan_position_results(self):
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
        self.assertTrue(type(rs) is Results)

        # Standard-ish params collapses to 2 clusters.
        f1 = ClusterPositionFilter(eps=0.025, height=100, width=100)
        self.assertEqual(f1.keep_indices(rs), [0, 3])

        # Small eps is 4 clusters.
        f2 = ClusterPositionFilter(eps=0.000015, height=100, width=100)
        self.assertEqual(f2.keep_indices(rs), [0, 3, 4, 5])

        # Large scale means 1 cluster
        f3 = ClusterPositionFilter(eps=0.025, height=1000, width=1000)
        self.assertEqual(f3.keep_indices(rs), [0])

        # Catch invalid parameters
        self.assertRaises(ValueError, ClusterPositionFilter, eps=0.025, height=100, width=0)
        self.assertRaises(ValueError, ClusterPositionFilter, eps=0.025, height=0, width=100)

    def test_dbscan_all(self):
        rs = self._make_result_list(
            [
                [10, 11, 1, 2],
                [10, 11, 1000, -1000],
                [10, 11, 1.0, 2.1],
                [55, 54, 1.0, 1.0],
                [55, 56, 10.0, 10.0],
                [10, 12, 4.1, 8],
            ]
        )
        self.assertTrue(type(rs) is ResultList)

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

    def test_dbscan_all_results(self):
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
        self.assertTrue(type(rs) is Results)

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
        rs = self._make_result_list(
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
        self.assertTrue(type(rs) is ResultList)

        f1 = ClusterMidPosFilter(eps=0.1, height=20, width=20, times=self.times)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 6])

        # Use different times.
        f2 = ClusterMidPosFilter(eps=0.1, height=20, width=20, times=[0, 10, 20])
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3, 4, 5, 6])

        # Use different times.
        f3 = ClusterMidPosFilter(eps=0.1, height=20, width=20, times=[0, 0.001, 0.002])
        self.assertEqual(f3.keep_indices(rs), [0, 3, 5])

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
        self.assertTrue(type(rs) is Results)

        f1 = ClusterMidPosFilter(eps=0.1, height=20, width=20, times=self.times)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 6])

        # Use different times.
        f2 = ClusterMidPosFilter(eps=0.1, height=20, width=20, times=[0, 10, 20])
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3, 4, 5, 6])

        # Use different times.
        f3 = ClusterMidPosFilter(eps=0.1, height=20, width=20, times=[0, 0.001, 0.002])
        self.assertEqual(f3.keep_indices(rs), [0, 3, 5])

    def test_clustering(self):
        cluster_params = {
            "ang_lims": [0.0, 1.5],
            "cluster_type": "all",
            "eps": 0.03,
            "mjd": self.times,
            "width": 100,
            "height": 100,
            "vel_lims": [5.0, 40.0],
        }

        results = self._make_result_list(
            [
                [10, 11, 1, 2],
                [10, 11, 10, 20],
                [40, 5, -1, 2],
                [5, 0, 1, 2],
                [5, 1, 1, 2],
            ]
        )
        self.assertTrue(type(results) is ResultList)
        apply_clustering(results, cluster_params)
        self.assertEqual(results.num_results(), 4)

        # Try clustering with only positions.
        results2 = self._make_result_list(
            [
                [10, 11, 1, 2],
                [10, 11, 10, 20],
                [40, 5, -1, 2],
                [5, 0, 1, 2],
                [5, 1, 1, 2],
            ]
        )
        self.assertTrue(type(results2) is ResultList)
        cluster_params["cluster_type"] = "position"
        apply_clustering(results2, cluster_params)
        self.assertEqual(results2.num_results(), 3)

        # Try invalid or missing cluster_type.
        cluster_params["cluster_type"] = "invalid"
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)

        del cluster_params["cluster_type"]
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)

    def test_clustering_results(self):
        cluster_params = {
            "ang_lims": [0.0, 1.5],
            "cluster_type": "all",
            "eps": 0.03,
            "mjd": self.times,
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
        self.assertTrue(type(results) is Results)
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
        self.assertTrue(type(results2) is Results)
        cluster_params["cluster_type"] = "position"
        apply_clustering(results2, cluster_params)
        self.assertEqual(len(results2), 3)

        # Try invalid or missing cluster_type.
        cluster_params["cluster_type"] = "invalid"
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)

        del cluster_params["cluster_type"]
        self.assertRaises(ValueError, apply_clustering, results2, cluster_params)


if __name__ == "__main__":
    unittest.main()
