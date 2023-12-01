import unittest

from kbmod.filters.stats_filters import *
from kbmod.result_list import *
from kbmod.search import *
from kbmod.trajectory_utils import make_trajectory


class test_basic_filters(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

        self.rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            trj = make_trajectory(lh=float(i))
            row = ResultRow(trj, self.num_times)
            row.filter_indices([k for k in range(i)])
            self.rs.append_result(row)

    def test_filter_min_likelihood(self):
        self.assertEqual(self.rs.num_results(), 10)

        f = LHFilter(4.5, None)
        self.assertEqual(f.get_filter_name(), "LH_Filter_4.5_to_None")

        # Do the filtering and check we have the correct ones.
        self.rs.apply_filter(f)
        self.assertEqual(self.rs.num_results(), 5)
        for i in range(self.rs.num_results()):
            self.assertGreater(self.rs.results[i].final_likelihood, 4.5)

        # Check the filtered results
        filtered = self.rs.get_filtered(f.get_filter_name())
        self.assertEqual(len(filtered), 5)
        for row in filtered:
            self.assertLess(row.final_likelihood, 4.5)

    def test_filter_max_likelihood(self):
        self.assertEqual(self.rs.num_results(), 10)

        f = LHFilter(None, 7.5)
        self.assertEqual(f.get_filter_name(), "LH_Filter_None_to_7.5")

        # Do the filtering and check we have the correct ones.
        self.rs.apply_filter(f)
        self.assertEqual(self.rs.num_results(), 8)
        for i in range(self.rs.num_results()):
            self.assertLess(self.rs.results[i].final_likelihood, 7.5)

        # Check the filtered results
        filtered = self.rs.get_filtered(f.get_filter_name())
        self.assertEqual(len(filtered), 2)
        for row in filtered:
            self.assertGreater(row.final_likelihood, 7.5)

    def test_filter_both_likelihood(self):
        self.assertEqual(self.rs.num_results(), 10)

        # Do the filtering and check we have the correct ones.
        self.rs.apply_filter(LHFilter(5.5, 7.5))
        self.assertEqual(self.rs.num_results(), 2)
        self.assertEqual(self.rs.results[0].final_likelihood, 6.0)
        self.assertEqual(self.rs.results[1].final_likelihood, 7.0)

    def test_filter_likelihood_mp(self):
        # Create a lot more results.
        rs = ResultList(self.times, track_filtered=True)
        for i in range(1000):
            trj = make_trajectory(lh=0.01 * float(i))
            row = ResultRow(trj, self.num_times)
            rs.append_result(row)
        self.assertEqual(rs.num_results(), 1000)

        # Do the filtering and check we have the correct ones.
        rs.apply_filter(LHFilter(5.5, 7.5), num_threads=10)
        self.assertEqual(rs.num_results(), 201)
        for row in rs.results:
            self.assertLessEqual(row.final_likelihood, 7.5)
            self.assertGreaterEqual(row.final_likelihood, 5.5)

        # Check the filtered results
        filtered = rs.get_filtered()
        self.assertEqual(len(filtered), 799)
        for row in filtered:
            self.assertTrue(row.final_likelihood > 7.5 or row.final_likelihood < 5.5)

    def test_filter_valid_indices(self):
        self.assertEqual(self.rs.num_results(), 10)

        f = NumObsFilter(4)
        self.assertEqual(f.get_filter_name(), "MinObsFilter_4")

        # Do the filtering and check we have the correct ones.
        self.rs.apply_filter(f)
        self.assertEqual(self.rs.num_results(), 6)
        for i in range(self.rs.num_results()):
            self.assertGreaterEqual(len(self.rs.results[i].valid_indices), 4)


if __name__ == "__main__":
    unittest.main()
