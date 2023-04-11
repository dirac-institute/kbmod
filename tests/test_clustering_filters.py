import unittest

from kbmod.filters.clustering_filters import *
from kbmod.result_list import ResultList, ResultRow
from kbmod.search import *


class test_clustering_filters(unittest.TestCase):
    def _make_trajectory(self, x0, y0, xv, yv, lh):
        t = trajectory()
        t.x = x0
        t.y = y0
        t.x_v = xv
        t.y_v = yv
        t.lh = lh
        return t

    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

        self.trjs = [
            self._make_trajectory(10, 11, 1, 2, 100.0),
            self._make_trajectory(10, 11, 10, 20, 100.0),
            self._make_trajectory(40, 5, -1, 2, 100.0),
            self._make_trajectory(5, 0, 1, 2, 100.0),
            self._make_trajectory(5, 1, 1, 2, 100.0),
            self._make_trajectory(5, 0, 5, -5, 100.0),
            self._make_trajectory(5, 1, 1.2, 2, 100.0),
            self._make_trajectory(5, 40, 5, -5, 100.0),
            self._make_trajectory(6, 40, 5, 5, 100.0),
            self._make_trajectory(20, 22, 5, 5, 100.0),
            self._make_trajectory(21, 23, 5, 5, 100.0),
        ]
        
        self.rs = ResultList(self.times, track_filtered=True)
        for t in self.trjs:
            row = ResultRow(t, self.num_times)
            self.rs.append_result(row)

    def test_dbscan_position(self):
        f = DBSCANFilter("position", 0.015, 100, 100, [0, 50], [0, 1.5], self.times)
        inds = f.keep_indices(self.rs)
        self.assertEqual(inds, [0, 2, 3, 7, 9])

    def test_dbscan_position_eps(self):
        f = DBSCANFilter("position", 0.2, 100, 100, [0, 50], [0, 1.5], self.times)
        inds = f.keep_indices(self.rs)
        self.assertEqual(inds, [0, 2, 7])

    def test_dbscan_position_scale(self):
        f = DBSCANFilter("position", 0.015, 10, 10, [0, 50], [0, 1.5], self.times)
        inds = f.keep_indices(self.rs)
	self.assertEqual(inds, [0, 2, 3, 4, 7, 8, 9, 10])

    def test_dbscan_all(self):
        f = DBSCANFilter("all", 0.05, 100, 100, [0, 100], [0, 3.0], self.times)
        inds = f.keep_indices(self.rs)
        self.assertEqual(inds, [0, 1, 2, 3, 5, 7, 8, 9])

    def test_dbscan_all_scale(self):
        f = DBSCANFilter("all", 0.05, 100, 100, [0, 20], [0, 1.5], self.times)
        inds = f.keep_indices(self.rs)
        self.assertEqual(inds, [0, 1, 2, 3, 5, 6, 7, 8, 9])

if __name__ == "__main__":
    unittest.main()
