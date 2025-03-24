import csv
import numpy as np
import os
import tempfile
import unittest

from astropy.table import Table
import os.path as ospath
from pathlib import Path

from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod.wcs_utils import make_fake_wcs, wcs_fits_equal


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
            trj = Trajectory(x=i, y=i + 0, vx=i - 2.0, vy=i + 5.0, flux=5.0 * i, lh=100.0 + i, obs_count=i)
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

    def test_empty(self):
        table = Results()
        self.assertEqual(len(table), 0)
        self.assertEqual(len(table.colnames), 7)
        self.assertEqual(table.get_num_times(), 0)
        self.assertIsNone(table.wcs)
        self.assertIsNone(table.times)

        # Check that we don't crash on updating the likelihoods.
        table._update_likelihood()

    def test_from_trajectories(self):
        table = Results.from_trajectories(self.trj_list)
        self.assertEqual(len(table), self.num_entries)
        self.assertEqual(len(table.colnames), 7)
        self.assertIsNone(table.wcs)
        self.assertIsNone(table.times)
        self._assert_results_match_dict(table, self.input_dict)

    def test_from_dict(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]

        # Missing 'x' column
        del self.input_dict["x"]
        with self.assertRaises(KeyError):
            _ = Results(self.input_dict)

        # Add back the 'x' column.
        self.input_dict["x"] = [trj.x for trj in self.trj_list]
        table = Results(self.input_dict)
        self._assert_results_match_dict(table, self.input_dict)

    def test_from_table(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]

        # Missing 'x' column
        del self.input_dict["x"]
        with self.assertRaises(KeyError):
            _ = Results(Table(self.input_dict))

        # Add back the 'x' column.
        self.input_dict["x"] = [trj.x for trj in self.trj_list]
        table = Results(Table(self.input_dict))
        self._assert_results_match_dict(table, self.input_dict)

    def test_copy(self):
        table1 = Results(self.input_dict)
        table2 = table1.copy()

        # Add a new column to table2 and check that it is not in table1
        # (i.e. we have done a deep copy).
        table2.table["something_added"] = [i for i in range(self.num_entries)]
        self.assertTrue("something_added" in table2.colnames)
        self.assertFalse("something_added" in table1.colnames)

    def test_make_trajectory_list(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]
        table = Results(self.input_dict)

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

    def test_remove_column(self):
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]
        table = Results(self.input_dict)
        self.assertTrue("something_added" in table.colnames)

        # Can't drop a column that is not there.
        with self.assertRaises(KeyError):
            table.remove_column("missing_column")

        table.remove_column("something_added")
        self.assertFalse("something_added" in table.colnames)

        # Can't drop a required column.
        with self.assertRaises(KeyError):
            table.remove_column("x")

    def test_extend(self):
        table1 = Results.from_trajectories(self.trj_list)
        for i in range(self.num_entries):
            self.trj_list[i].x += self.num_entries
        table2 = Results.from_trajectories(self.trj_list)

        table1.extend(table2)
        self.assertEqual(len(table1), 2 * self.num_entries)
        for i in range(2 * self.num_entries):
            self.assertEqual(table1["x"][i], i)

        # Fail with a mismatched table.
        self.input_dict["something_added"] = [i for i in range(self.num_entries)]
        table3 = Results(self.input_dict)
        with self.assertRaises(ValueError):
            table1.extend(table3)

        # Test starting from an empty table.
        table4 = Results()
        table4.extend(table1)
        self.assertEqual(len(table1), len(table4))
        for i in range(self.num_entries):
            self.assertEqual(table1["x"][i], i)

    def test_add_psi_phi(self):
        num_to_use = 3
        table = Results.from_trajectories(self.trj_list[0:num_to_use])
        psi_array = np.array([[1.0, 1.1, 1.2, 1.3] for i in range(num_to_use)])
        phi_array = np.array([[1.0, 1.0, 0.0, 2.0] for i in range(num_to_use)])
        obs_valid = np.array(
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
        table.add_psi_phi_data(psi_array, phi_array, obs_valid)
        for i in range(num_to_use):
            self.assertEqual(len(table["psi_curve"][i]), 4)
            self.assertEqual(len(table["phi_curve"][i]), 4)
            self.assertEqual(len(table["obs_valid"][i]), 4)

            self.assertAlmostEqual(table["likelihood"][i], exp_lh[i], delta=1e-5)
            self.assertAlmostEqual(table["flux"][i], exp_flux[i], delta=1e-5)
            self.assertEqual(table["obs_count"][i], exp_obs[i])
        self.assertEqual(table.get_num_times(), 4)

    def test_update_obs_valid(self):
        num_to_use = 3
        table = Results.from_trajectories(self.trj_list[0:num_to_use])
        psi_array = np.array([[1.0, 1.1, 1.2, 1.3] for i in range(num_to_use)])
        phi_array = np.array([[1.0, 1.0, 0.0, 2.0] for i in range(num_to_use)])
        table.add_psi_phi_data(psi_array, phi_array)
        for i in range(num_to_use):
            self.assertAlmostEqual(table["likelihood"][i], 2.3, delta=1e-5)
            self.assertAlmostEqual(table["flux"][i], 1.15, delta=1e-5)
            self.assertEqual(table["obs_count"][i], 4)

        # Add the obs_valid column later to simulate sigmaG clipping.
        obs_valid = np.array(
            [
                [True, True, True, True],
                [True, False, True, True],
                [False, False, False, False],
            ]
        )
        table.update_obs_valid(obs_valid, drop_empty_rows=False)
        self.assertEqual(len(table), 3)
        self.assertEqual(table.get_num_times(), 4)

        exp_lh = [2.3, 2.020725, 0.0]
        exp_flux = [1.15, 1.1666667, 0.0]
        exp_obs = [4, 3, 0]
        for i in range(num_to_use):
            self.assertEqual(len(table["obs_valid"][i]), 4)
            self.assertAlmostEqual(table["likelihood"][i], exp_lh[i], delta=1e-5)
            self.assertAlmostEqual(table["flux"][i], exp_flux[i], delta=1e-5)
            self.assertEqual(table["obs_count"][i], exp_obs[i])

        # Check that when drop_empty_rows is set, we filter the rows with no observations.
        table.update_obs_valid(obs_valid, drop_empty_rows=True)
        self.assertEqual(len(table), 2)

    def test_compute_likelihood_curves(self):
        num_to_use = 3
        table = Results.from_trajectories(self.trj_list[0:num_to_use])

        psi_array = np.array(
            [
                [1.0, 1.1, 1.0, 1.3],
                [10.0, np.nan, np.inf, 1.3],
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
        obs_valid = np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, True],
            ]
        )
        table.add_psi_phi_data(psi_array, phi_array, obs_valid)

        expected1 = np.array([[1.0, 1.1, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 1.0, 5.0, 0.25]])
        lh_mat1 = table.compute_likelihood_curves(filter_obs=False)
        self.assertTrue(np.allclose(lh_mat1, expected1))

        expected2 = np.array([[1.0, 1.1, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [0.2, 1.0, 0.0, 0.25]])
        lh_mat2 = table.compute_likelihood_curves(filter_obs=True)
        self.assertTrue(np.allclose(lh_mat2, expected2))

        # Try masking with NAN. This replaces ALL the invalid cells.
        lh_mat3 = table.compute_likelihood_curves(filter_obs=True, mask_value=np.nan)
        expected = np.array(
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, True],
            ]
        )
        self.assertTrue(np.array_equal(np.isfinite(lh_mat3), expected))

    def test_is_empty_value(self):
        table = Results.from_trajectories(self.trj_list)

        # Create a two new columns: one with integers and the other with meaningless
        # index pairs (three of which are empty)
        nums_col = [i for i in range(len(table))]
        table.table["nums"] = nums_col

        pairs_col = [(i, i + 1) for i in range(len(table))]
        pairs_col[1] = None
        pairs_col[3] = ()
        pairs_col[7] = ()
        table.table["pairs"] = pairs_col

        expected = [False] * len(table)
        expected[1] = True
        expected[3] = True
        expected[7] = True

        # Check that we can tell which entries are empty.
        nums_is_empty = table.is_empty_value("nums")
        self.assertFalse(np.any(nums_is_empty))

        pairs_is_empty = table.is_empty_value("pairs")
        self.assertTrue(np.array_equal(pairs_is_empty, expected))

    def test_filter_by_index(self):
        table = Results.from_trajectories(self.trj_list)
        self.assertEqual(len(table), self.num_entries)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7]
        table.filter_rows(inds, "index_test")
        self.assertEqual(len(table), len(inds))
        for i in range(len(inds)):
            self.assertEqual(table["x"][i], self.trj_list[inds[i]].x)

        # Check that we record the stats of the filtered even if we are not
        # keeping the full tables.
        self.assertTrue("index_test" in table.filtered_stats)
        self.assertEqual(table.filtered_stats["index_test"], self.num_entries - len(inds))

        # Without tracking there should be nothing stored in the Results's
        # filtered dictionary.
        self.assertFalse("index_test" in table.filtered)
        self.assertEqual(len(table.filtered), 0)
        with self.assertRaises(ValueError):
            table.get_filtered()

        # Without tracking we cannot revert anything.
        with self.assertRaises(ValueError):
            table.revert_filter()

    def test_filter_by_mask(self):
        table = Results.from_trajectories(self.trj_list)
        self.assertEqual(len(table), self.num_entries)

        # Do the filtering and check we have the correct ones.
        mask = [False] * self.num_entries
        inds = [0, 2, 6, 7]
        for i in inds:
            mask[i] = True

        table.filter_rows(mask, "mask_test")
        self.assertEqual(len(table), len(inds))
        for i in range(len(inds)):
            self.assertEqual(table["x"][i], self.trj_list[inds[i]].x)

        # Check that we record the stats of the filtered even if we are not
        # keeping the full tables.
        self.assertTrue("mask_test" in table.filtered_stats)
        self.assertEqual(table.filtered_stats["mask_test"], self.num_entries - len(inds))

    def test_filter_empty(self):
        table = Results.from_trajectories([])
        self.assertEqual(len(table), 0)

        # Do the filtering and check we have the correct ones.
        table.filter_rows([], "empty_test")
        self.assertEqual(len(table), 0)
        self.assertTrue("empty_test" in table.filtered_stats)

    def test_filter_by_index_tracked(self):
        table = Results.from_trajectories(self.trj_list[0:10], track_filtered=True)
        self.assertEqual(len(table), 10)

        # Do the filtering. First remove elements 0 and 2. Then remove elements
        # 0, 5, and 6 from the resulting list (1, 7, 8 in the original list).
        table.filter_rows([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        self.assertEqual(len(table), 8)
        table.filter_rows([1, 2, 3, 4, 7], label="filter2")
        self.assertEqual(len(table), 5)
        self.assertEqual(table["x"][0], 3)
        self.assertEqual(table["x"][1], 4)
        self.assertEqual(table["x"][2], 5)
        self.assertEqual(table["x"][3], 6)
        self.assertEqual(table["x"][4], 9)

        # Check that we can get the correct filtered counts.
        self.assertEqual(table.filtered_stats["filter1"], 2)
        self.assertEqual(table.filtered_stats["filter2"], 3)

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
        self.assertFalse("filter2" in table.filtered_stats)

        # Check that we can revert the filtering and add a 'filtered_reason' column.
        table = Results.from_trajectories(self.trj_list[0:10], track_filtered=True)
        table.filter_rows([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        table.filter_rows([1, 2, 3, 4, 7], label="filter2")
        table.revert_filter(add_column="reason")
        self.assertEqual(len(table), 10)
        expected_order = [3, 4, 5, 6, 9, 0, 2, 1, 7, 8]
        expected_reason = ["", "", "", "", "", "filter1", "filter1", "filter2", "filter2", "filter2"]
        for i, value in enumerate(expected_order):
            self.assertEqual(table["x"][i], value)
            self.assertEqual(table["reason"][i], expected_reason[i])

    def test_extend_with_filtered(self):
        table1 = Results.from_trajectories(self.trj_list, track_filtered=True)
        for i in range(self.num_entries):
            self.trj_list[i].x += self.num_entries
        table2 = Results.from_trajectories(self.trj_list, track_filtered=True)

        table1.filter_rows([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        table1.filter_rows([1, 2, 3, 4, 7], label="filter2")
        table2.filter_rows([1, 3, 4, 5, 6, 7, 8], label="filter1")
        table2.filter_rows([1], label="filter3")

        table1.extend(table2)
        self.assertEqual(len(table1), 6)
        self.assertEqual(table1.filtered_stats["filter1"], 5)
        self.assertEqual(table1.filtered_stats["filter2"], 3)
        self.assertEqual(table1.filtered_stats["filter3"], 6)
        self.assertEqual(len(table1.get_filtered("filter1")), 5)
        self.assertEqual(len(table1.get_filtered("filter2")), 3)
        self.assertEqual(len(table1.get_filtered("filter3")), 6)

    def test_to_from_table_file(self):
        max_save = 5
        table = Results.from_trajectories(self.trj_list[0:max_save], track_filtered=True)
        table.table["other"] = [i for i in range(max_save)]
        self.assertEqual(len(table), max_save)

        # Create a fake WCS to use for serialization tests.
        fake_wcs = make_fake_wcs(25.0, -7.5, 800, 600, deg_per_pixel=0.01)
        table.wcs = fake_wcs

        # Add fake times.
        table.times = np.array([1, 2, 3, 4, 5])

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

            # Check that we reloaded the WCS's, including the correct shape.
            self.assertIsNotNone(table2.wcs)
            self.assertTrue(wcs_fits_equal(table2.wcs, fake_wcs))
            self.assertEqual(table2.wcs.pixel_shape, fake_wcs.pixel_shape)

            # Cannot overwrite with it set to False
            with self.assertRaises(OSError):
                table.write_table(file_path, overwrite=False, cols_to_drop=["other"])

            # We can overwrite with droped columns and additional meta data.
            table.write_table(
                file_path,
                overwrite=True,
                cols_to_drop=["other"],
                extra_meta={"other": 100.0},
            )

            table3 = Results.read_table(file_path)
            self.assertEqual(len(table2), max_save)

            # We only dropped the table from the save file.
            self.assertFalse("other" in table3.colnames)
            self.assertTrue("other" in table.colnames)

            # We saved the additional meta data, including the WCS.
            self.assertTrue(np.array_equal(table3.table.meta["times"], [1, 2, 3, 4, 5]))
            self.assertEqual(table3.table.meta["other"], 100.0)
            self.assertIsNotNone(table3.wcs)
            self.assertTrue(wcs_fits_equal(table3.wcs, fake_wcs))

    def test_write_and_load_column(self):
        table = Results.from_trajectories(self.trj_list)
        self.assertFalse("all_stamps" in table.colnames)

        # Create a table with an extra column.
        table2 = Results.from_trajectories(self.trj_list)
        all_stamps = []
        for i in range(len(table)):
            all_stamps.append([np.full((5, 5), i), np.full((5, 5), i + 10)])
        table2.table["all_stamps"] = all_stamps
        self.assertTrue("all_stamps" in table2.colnames)

        # Try outputting the ResultList
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "all_stamps.npy")
            self.assertFalse(Path(file_path).is_file())

            # Can't load if the file is not there.
            with self.assertRaises(FileNotFoundError):
                table.load_column(file_path, "all_stamps")

            table2.write_column("all_stamps", file_path)
            self.assertTrue(Path(file_path).is_file())

            # Load the results into a new data structure and confirm they match.
            table.load_column(file_path, "all_stamps")
            self.assertTrue("all_stamps" in table.colnames)
            for i in range(len(table2)):
                self.assertEqual(table["all_stamps"][i].shape, (2, 5, 5))
                self.assertEqual(table["all_stamps"][i][0][0][0], i)
                self.assertEqual(table["all_stamps"][i][1][0][0], i + 10)

            # Change the number of rows and resave.
            table2.filter_rows([0, 1, 2])
            table2.write_column("all_stamps", file_path)

            # Loading to table 1 should now give a size mismatch error.
            with self.assertRaises(ValueError):
                table.load_column(file_path, "all_stamps_smaller")

    def test_write_filter_stats(self):
        table = Results.from_trajectories(self.trj_list)
        table.filter_rows([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        table.filter_rows([1, 2, 3, 4, 7], label="filter2")

        # Try outputting the ResultList
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "filtered_stats.csv")
            table.write_filtered_stats(file_path)

            # Read in the CSV file to a list of lists.
            data = []
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    data.append(row)

            # Check that the data matches.
            self.assertEqual(data[0][0], "unfiltered")
            self.assertEqual(data[0][1], "5")
            self.assertEqual(data[1][0], "filter1")
            self.assertEqual(data[1][1], "2")
            self.assertEqual(data[2][0], "filter2")
            self.assertEqual(data[2][1], "3")

    def test_mask_based_on_invalid_obs(self):
        num_times = 5
        mjds = np.array([i for i in range(num_times)])

        num_results = 6
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
                [False, False, True, False, False],
            ]
        )
        table.update_obs_valid(obs_valid)

        data_mat = np.full((num_results, num_times), 1.0)
        masked_mat = table.mask_based_on_invalid_obs(data_mat, -1.0)
        for r in range(num_results):
            for c in range(num_times):
                if obs_valid[r][c]:
                    self.assertEqual(masked_mat[r][c], 1.0)
                else:
                    self.assertEqual(masked_mat[r][c], -1.0)


if __name__ == "__main__":
    unittest.main()
