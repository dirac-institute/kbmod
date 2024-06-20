import numpy as np
import unittest

from kbmod.filters.basic_filters import apply_likelihood_clipping, apply_time_range_filter
from kbmod.results import Results
from kbmod.search import Trajectory


class test_basic_filters(unittest.TestCase):
    def test_apply_likelihood_clipping(self):
        num_times = 5
        num_results = 3
        trj_all = [Trajectory() for _ in range(num_results)]
        table = Results.from_trajectories(trj_all)

        psi_all = np.array([[1, 2, 3, 0, 5], [10, -1, 3, 4, 5], [1, 2, 100, 3, -10]])
        phi_all = np.full((num_results, num_times), 1.0)
        table.add_psi_phi_data(psi_all, phi_all)

        apply_likelihood_clipping(table, 1.0, 5.0)

        expected = [
            [True, True, True, False, True],
            [False, False, True, True, True],
            [True, True, False, True, False],
        ]
        for i in range(num_results):
            for j in range(num_times):
                self.assertEqual(expected[i][j], table["obs_valid"][i][j])

    def test_apply_time_range_filter(self):
        num_times = 5
        mjds = np.array([i for i in range(num_times)])

        num_results = 7
        trj_all = [Trajectory(x=i) for i in range(num_results)]
        table = Results.from_trajectories(trj_all)
        self.assertEqual(len(table), num_results)

        obs_valid = np.array(
            [
                [True, True, True, False, True],
                [True, True, True, True, False],
                [False, False, True, True, True],
                [False, True, True, True, False],
                [True, False, False, False, True],
                [False, False, False, True, False],
                [False, False, True, False, False],
            ]
        )
        table.update_obs_valid(obs_valid)

        self.assertFalse("duration" in table.colnames)
        apply_time_range_filter(table, mjds, 3, colname="duration")
        self.assertTrue("duration" in table.colnames)

        self.assertEqual(len(table), 3)
        self.assertEqual(table["x"][0], 0)
        self.assertEqual(table["x"][1], 1)
        self.assertEqual(table["x"][2], 4)
        self.assertEqual(table["duration"][0], 4.0)
        self.assertEqual(table["duration"][1], 3.0)
        self.assertEqual(table["duration"][2], 4.0)


if __name__ == "__main__":
    unittest.main()
