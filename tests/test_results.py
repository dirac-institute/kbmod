import numpy as np
import os
import tempfile
import unittest

from astropy.table import Table
from pathlib import Path

from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod.trajectory_utils import make_trajectory


class test_results(unittest.TestCase):
    def setUp(self):
        self.num_entries = 10
        self.input_dict = {
            "x": [],
            "y": [],
            "vx": [],
            "vy": [],
            "flux": [],
            "likelihood": [],
            "obs_count": [],
        }
        self.trj_list = []

        for i in range(self.num_entries):
            trj = make_trajectory(
                x=i, y=i + 0, vx=i - 2.0, vy=i + 5.0, flux=5.0 * i, lh=100.0 + i, obs_count=i
            )
            self.trj_list.append(trj)
            self.input_dict["x"].append(trj.x)
            self.input_dict["y"].append(trj.y)
            self.input_dict["vx"].append(trj.vx)
            self.input_dict["vy"].append(trj.vy)
            self.input_dict["flux"].append(trj.flux)
            self.input_dict["likelihood"].append(trj.lh)
            self.input_dict["obs_count"].append(trj.obs_count)

    def _assert_results_match_dict(self, results, test_dict):
        # Check that the shape of the results are the same.
        self.assertEqual(len(results), len(test_dict["x"]))
        self.assertEqual(set(results.colnames), set(test_dict.keys()))

        for col in results.colnames:
            for i in range(len(results)):
                self.assertEqual(results[col][i], test_dict[col][i])

    def test_create(self):
        table = Results(self.trj_list)
        self.assertEqual(len(table), self.num_entries)
        self.assertEqual(len(table.colnames), 7)
        self._assert_results_match_dict(table, self.input_dict)

        # Test that we ignore invalid results, but track them in the filtered table.
        self.trj_list[2].valid = False
        self.trj_list[7].valid = False
        table2 = Results(self.trj_list, track_filtered=True)
        self.assertEqual(len(table2), self.num_entries - 2)
        for i in range(self.num_entries - 2):
            self.assertFalse(table2["x"][i] == 2 or table2["x"][i] == 7)

        filtered = table2.get_filtered()
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered["x"][0], 2)
        self.assertEqual(filtered["x"][1], 7)

    def test_from_dict(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]

        # Missing 'x' column
        del self.input_dict["x"]
        with self.assertRaises(KeyError):
            _ = Results.from_dict(self.input_dict)

        # Add back the 'x' column.
        self.input_dict["x"] = [trj.x for trj in self.trj_list]
        table = Results.from_dict(self.input_dict)
        self._assert_results_match_dict(table, self.input_dict)

    def test_from_table(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]

        # Missing 'x' column
        del self.input_dict["x"]
        with self.assertRaises(KeyError):
            _ = Results.from_table(Table(self.input_dict))

        # Add back the 'x' column.
        self.input_dict["x"] = [trj.x for trj in self.trj_list]
        table = Results.from_table(Table(self.input_dict))
        self._assert_results_match_dict(table, self.input_dict)

    def test_make_trajectory_list(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]
        table = Results.from_dict(self.input_dict)

        trajectories = table.make_trajectory_list()
        self.assertEqual(len(trajectories), self.num_entries)
        for i, trj in enumerate(trajectories):
            self.assertEqual(trj.x, table["x"][i])
            self.assertEqual(trj.y, table["y"][i])
            self.assertEqual(trj.vx, table["vx"][i])
            self.assertEqual(trj.vy, table["vy"][i])
            self.assertEqual(trj.obs_count, table["obs_count"][i])
            self.assertEqual(trj.flux, table["flux"][i])
            self.assertEqual(trj.lh, table["likelihood"][i])

    def test_extend(self):
        table1 = Results(self.trj_list)
        for i in range(self.num_entries):
            self.trj_list[i].x += self.num_entries
        table2 = Results(self.trj_list)

        table1.extend(table2)
        self.assertEqual(len(table1), 2 * self.num_entries)
        for i in range(2 * self.num_entries):
            self.assertEqual(table1["x"][i], i)

        # Fail with a mismatched table.
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]
        table3 = Results.from_dict(self.input_dict)
        with self.assertRaises(ValueError):
            table1.extend(table3)

    def test_add_psi_phi(self):
        num_to_use = 3
        table = Results(self.trj_list[0:num_to_use])
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

        # Check the the data has been inserted and the statistics have been updated.
        table.add_psi_phi_data(psi_array, phi_array, index_valid)
        for i in range(num_to_use):
            self.assertEqual(len(table["psi_curve"][i]), 4)
            self.assertEqual(len(table["phi_curve"][i]), 4)
            self.assertEqual(len(table["index_valid"][i]), 4)

            self.assertAlmostEqual(table["likelihood"][i], exp_lh[i], delta=1e-5)
            self.assertAlmostEqual(table["flux"][i], exp_flux[i], delta=1e-5)
            self.assertEqual(table["obs_count"][i], exp_obs[i])

    def test_compute_likelihood_curves(self):
        num_to_use = 3
        table = Results(self.trj_list[0:num_to_use])

        psi_array = np.array(
            [
                [1.0, 1.1, 1.0, 1.3],
                [10.0, np.NAN, np.inf, 1.3],
                [1.0, 4.0, 10.0, 1.0],
            ]
        )
        phi_array = np.array(
            [
                [1.0, 1.0, 4.0, 0.0],
                [100.0, 10.0, 1.0, np.inf],
                [25.0, 16.0, 4.0, 16.0],
            ]
        )
        index_valid = np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, True],
            ]
        )
        table.add_psi_phi_data(psi_array, phi_array, index_valid)

        expected1 = np.array([[1.0, 1.1, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 1.0, 5.0, 0.25]])
        lh_mat1 = table.compute_likelihood_curves(filter_indices=False)
        self.assertTrue(np.allclose(lh_mat1, expected1))

        expected2 = np.array([[1.0, 1.1, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 1.0, 0.0, 0.25]])
        lh_mat2 = table.compute_likelihood_curves(filter_indices=True)
        self.assertTrue(np.allclose(lh_mat2, expected2))

    def test_filter_by_index(self):
        table = Results(self.trj_list)
        self.assertEqual(len(table), self.num_entries)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7]
        table.filter_by_index(inds)
        self.assertEqual(len(table), len(inds))
        for i in range(len(inds)):
            self.assertEqual(table["x"][i], self.trj_list[inds[i]].x)

        # Without tracking there should be nothing stored in the Results's
        # filtered dictionary.
        self.assertEqual(len(table.filtered), 0)
        with self.assertRaises(ValueError):
            table.get_filtered()

        # Without tracking we cannot revert anything.
        with self.assertRaises(ValueError):
            table.revert_filter()

    def test_filter_by_index_tracked(self):
        table = Results(self.trj_list[0:10], track_filtered=True)
        self.assertEqual(len(table), 10)

        # Do the filtering. First remove elements 0 and 2. Then remove elements
        # 0, 5, and 6 from the resulting list (1, 7, 8 in the original list).
        table.filter_by_index([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        self.assertEqual(len(table), 8)
        table.filter_by_index([1, 2, 3, 4, 7], label="filter2")
        self.assertEqual(len(table), 5)
        self.assertEqual(table["x"][0], 3)
        self.assertEqual(table["x"][1], 4)
        self.assertEqual(table["x"][2], 5)
        self.assertEqual(table["x"][3], 6)
        self.assertEqual(table["x"][4], 9)

        # Check that we can get the correct filtered rows.
        f1 = table.get_filtered("filter1")
        self.assertEqual(len(f1), 2)
        self.assertEqual(f1["x"][0], 0)
        self.assertEqual(f1["x"][1], 2)

        f2 = table.get_filtered("filter2")
        self.assertEqual(len(f2), 3)
        self.assertEqual(f2["x"][0], 1)
        self.assertEqual(f2["x"][1], 7)
        self.assertEqual(f2["x"][2], 8)

        # Check that not passing a label gives us all filtered results.
        f_all = table.get_filtered()
        self.assertEqual(len(f_all), 5)

        # Check that we can revert the filtering.
        table.revert_filter("filter2")
        self.assertEqual(len(table), 8)
        expected_order = [3, 4, 5, 6, 9, 1, 7, 8]
        for i, value in enumerate(expected_order):
            self.assertEqual(table["x"][i], value)

        # Check that we can revert the filtering and add a 'filtered_reason' column.
        table = Results(self.trj_list[0:10], track_filtered=True)
        table.filter_by_index([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        table.filter_by_index([1, 2, 3, 4, 7], label="filter2")
        table.revert_filter(add_column="reason")
        self.assertEqual(len(table), 10)
        expected_order = [3, 4, 5, 6, 9, 0, 2, 1, 7, 8]
        expected_reason = ["", "", "", "", "", "filter1", "filter1", "filter2", "filter2", "filter2"]
        for i, value in enumerate(expected_order):
            self.assertEqual(table["x"][i], value)
            self.assertEqual(table["reason"][i], expected_reason[i])

    def test_to_from_table_file(self):
        max_save = 5
        table = Results(self.trj_list[0:max_save], track_filtered=True)
        table.table["other"] = [i for i in range(max_save)]
        self.assertEqual(len(table), max_save)

        # Test read/write to file.
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "results.ecsv")
            self.assertFalse(Path(file_path).is_file())

            table.write_table(file_path)
            self.assertTrue(Path(file_path).is_file())

            table2 = Results.read_table(file_path)
            self.assertEqual(len(table2), max_save)
            self.assertTrue("other" in table2.colnames)
            for col in ["x", "y", "vx", "vy", "likelihood", "flux", "obs_count", "other"]:
                self.assertTrue(np.allclose(table[col], table2[col]))

            # Cannot overwrite with it set to False
            with self.assertRaises(OSError):
                table.write_table(file_path, overwrite=False, cols_to_drop=["other"])

            # We can overwrite with droped columns
            table.write_table(file_path, overwrite=True, cols_to_drop=["other"])

            table3 = Results.read_table(file_path)
            self.assertEqual(len(table2), max_save)

            # We only dropped the table from the save file.
            self.assertFalse("other" in table3.colnames)
            self.assertTrue("other" in table.colnames)


if __name__ == "__main__":
    unittest.main()