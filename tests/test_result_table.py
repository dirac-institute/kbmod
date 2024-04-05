import numpy as np
import unittest

from astropy.table import Table

from kbmod.result_table import ResultTable
from kbmod.search import Trajectory
from kbmod.trajectory_utils import make_trajectory


class test_result_table(unittest.TestCase):
    def setUp(self):
        self.num_entries = 10
        self.trj_list = [
            make_trajectory(x=i, y=i + 0, vx=i - 2.0, vy=i + 5.0, flux=5.0 * i, lh=100.0 + i, obs_count=i)
            for i in range(self.num_entries)
        ]

    def test_create(self):
        table = ResultTable(self.trj_list)
        self.assertEqual(len(table), self.num_entries)
        for i in range(self.num_entries):
            self.assertEqual(table.results["x"][i], self.trj_list[i].x)
            self.assertEqual(table.results["y"][i], self.trj_list[i].y)
            self.assertEqual(table.results["vx"][i], self.trj_list[i].vx)
            self.assertEqual(table.results["vy"][i], self.trj_list[i].vy)
            self.assertEqual(table.results["flux"][i], self.trj_list[i].flux)
            self.assertEqual(table.results["likelihood"][i], self.trj_list[i].lh)
            self.assertEqual(table.results["obs_count"][i], self.trj_list[i].obs_count)
            self.assertTrue(type(table.results["trajectory"][i]) is Trajectory)

        # Test that we ignore invalid results.
        self.trj_list[2].valid = False
        self.trj_list[7].valid = False
        table2 = ResultTable(self.trj_list)
        self.assertEqual(len(table2), self.num_entries - 2)
        for i in range(self.num_entries - 2):
            self.assertFalse(table2.results["x"][i] == 2 or table2.results["x"][i] == 7)

    def test_extend(self):
        table1 = ResultTable(self.trj_list)
        for i in range(self.num_entries):
            self.trj_list[i].x += self.num_entries
        table2 = ResultTable(self.trj_list)

        table1.extend(table2)
        self.assertEqual(len(table1), 2 * self.num_entries)
        for i in range(2 * self.num_entries):
            self.assertEqual(table1.results["x"][i], i)

    def test_add_psi_phi(self):
        num_to_use = 3
        table = ResultTable(self.trj_list[0:num_to_use])
        psi_array = np.array([[1.0, 1.1, 1.2, 1.3] for i in range(num_to_use)])
        phi_array = np.array([[1.0, 1.0, 0.0, 2.0] for i in range(num_to_use)])
        index_valid = np.array(
            [
                [True, True, True, True],
                [True, False, True, True],
                [False, False, False, False],
            ]
        )

        exp_lh = [2.3, 2.020725, 0.0]
        exp_flux = [1.15, 1.1666667, 0.0]
        exp_obs = [4, 3, 0]

        # Check the the data has been inserted and the statistics have been updated
        # in both the columns and the Trajectory objects.
        table.add_psi_phi_data(psi_array, phi_array, index_valid)
        for i in range(num_to_use):
            self.assertEqual(len(table.results["psi_curve"][i]), 4)
            self.assertEqual(len(table.results["phi_curve"][i]), 4)
            self.assertEqual(len(table.results["index_valid"][i]), 4)

            self.assertAlmostEqual(table.results["likelihood"][i], exp_lh[i], delta=1e-5)
            self.assertAlmostEqual(table.results["flux"][i], exp_flux[i], delta=1e-5)
            self.assertEqual(table.results["obs_count"][i], exp_obs[i])

            self.assertAlmostEqual(table.results["trajectory"][i].lh, exp_lh[i], delta=1e-5)
            self.assertAlmostEqual(table.results["trajectory"][i].flux, exp_flux[i], delta=1e-5)
            self.assertEqual(table.results["trajectory"][i].obs_count, exp_obs[i])

    def test_filter_by_index(self):
        table = ResultTable(self.trj_list)
        self.assertEqual(len(table), self.num_entries)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7]
        table.filter_by_index(inds)
        self.assertEqual(len(table), len(inds))
        for i in range(len(inds)):
            self.assertEqual(table.results["x"][i], self.trj_list[inds[i]].x)

        # Without tracking there should be nothing stored in the ResultTables's
        # filtered dictionary.
        self.assertEqual(len(table.filtered), 0)
        with self.assertRaises(ValueError):
            table.get_filtered()

        # Without tracking we cannot revert anything.
        with self.assertRaises(ValueError):
            table.revert_filter()

    def test_filter_by_index_tracked(self):
        table = ResultTable(self.trj_list[0:10], track_filtered=True)
        self.assertEqual(len(table), 10)

        # Do the filtering. First remove elements 0 and 2. Then remove elements
        # 0, 5, and 6 from the resulting list (1, 7, 8 in the original list).
        table.filter_by_index([1, 3, 4, 5, 6, 7, 8, 9], label="1")
        self.assertEqual(len(table), 8)
        table.filter_by_index([1, 2, 3, 4, 7], label="2")
        self.assertEqual(len(table), 5)
        self.assertEqual(table.results["x"][0], 3)
        self.assertEqual(table.results["x"][1], 4)
        self.assertEqual(table.results["x"][2], 5)
        self.assertEqual(table.results["x"][3], 6)
        self.assertEqual(table.results["x"][4], 9)

        # Check that we can get the correct filtered rows.
        f1 = table.get_filtered("1")
        self.assertEqual(len(f1), 2)
        self.assertEqual(f1["x"][0], 0)
        self.assertEqual(f1["x"][1], 2)

        f2 = table.get_filtered("2")
        self.assertEqual(len(f2), 3)
        self.assertEqual(f2["x"][0], 1)
        self.assertEqual(f2["x"][1], 7)
        self.assertEqual(f2["x"][2], 8)

        # Check that not passing a label gives us all filtered results.
        f_all = table.get_filtered()
        self.assertEqual(len(f_all), 5)

        # Check that we can revert the filtering.
        table.revert_filter("2")
        self.assertEqual(len(table), 8)
        expected_order = [3, 4, 5, 6, 9, 1, 7, 8]
        for i, value in enumerate(expected_order):
            self.assertEqual(table.results["x"][i], value)


if __name__ == "__main__":
    unittest.main()
