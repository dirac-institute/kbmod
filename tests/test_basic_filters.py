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


if __name__ == "__main__":
    unittest.main()
