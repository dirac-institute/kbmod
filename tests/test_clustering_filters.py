import unittest

from kbmod.filters.clustering_filters import *
from kbmod.result_list import ResultList, ResultRow
from kbmod.search import *


class test_clustering_filters(unittest.TestCase):
    def _make_data(self, objs):
        """Create a ResultList for the given objects.

        Parameters
        ----------
        obj : list of lists
            A list where each element specifies a trajectory
            as [x, y, xv, yv].

        Returns
        -------
        ResultList
        """
        rs = ResultList(self.times, track_filtered=True)
        for x in objs:
            t = trajectory()
            t.x = x[0]
            t.y = x[1]
            t.x_v = x[2]
            t.y_v = x[3]
            t.lh = 100.0

            row = ResultRow(t, self.num_times)
            rs.append_result(row)
        return rs

    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

    def test_dbscan_position(self):
        rs = self._make_data(
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
        f1 = DBSCANFilter("position", 0.025, 100, 100, [0, 50], [0, 1.5], self.times)
        self.assertEqual(f1.keep_indices(rs), [0, 3])

        # Small eps is 4 clusters.
        f2 = DBSCANFilter("position", 0.000015, 100, 100, [0, 50], [0, 1.5], self.times)
        self.assertEqual(f2.keep_indices(rs), [0, 3, 4, 5])

        # Large scale means 1 cluster
        f3 = DBSCANFilter("position", 0.025, 1000, 1000, [0, 50], [0, 1.5], self.times)
        self.assertEqual(f3.keep_indices(rs), [0])

    def test_dbscan_all(self):
        rs = self._make_data(
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
        f1 = DBSCANFilter("all", 0.025, 100, 100, [0, 100], [0, 1.5], self.times)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 4, 5])

        # Larger eps is 3 clusters.
        f2 = DBSCANFilter("all", 0.25, 100, 100, [0, 100], [0, 1.5], self.times)
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3])

        # Larger scale is 3 clusters.
        f3 = DBSCANFilter("all", 0.025, 100, 100, [0, 5000], [0, 1.5], self.times)
        self.assertEqual(f3.keep_indices(rs), [0, 1, 3])

    def test_dbscan_mid_pos(self):
        rs = self._make_data(
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

        f1 = DBSCANFilter("mid_position", 0.1, 20, 20, [0, 100], [0, 1.5], self.times)
        self.assertEqual(f1.keep_indices(rs), [0, 1, 3, 6])

        # Use different times.
        f2 = DBSCANFilter("mid_position", 0.1, 20, 20, [0, 100], [0, 1.5], [0, 10, 20])
        self.assertEqual(f2.keep_indices(rs), [0, 1, 3, 4, 5, 6])

        # Use different times.
        f3 = DBSCANFilter("mid_position", 0.1, 20, 20, [0, 100], [0, 1.5], [0, 0.001, 0.002])
        self.assertEqual(f3.keep_indices(rs), [0, 3, 5])

if __name__ == "__main__":
    unittest.main()
