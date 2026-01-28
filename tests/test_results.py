import csv
import numpy as np
import os
import tempfile
import unittest

from astropy.table import Table
from pathlib import Path

from kbmod.results import Results, write_results_to_files_destructive
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
            "uuid": [],
        }
        self.trj_list = []

        for i in range(self.num_entries):
            trj = Trajectory(
                x=i,
                y=i + 0,
                vx=i - 2.0,
                vy=i + 5.0,
                flux=5.0 * i,
                lh=100.0 + i,
                obs_count=i,
            )
            self.trj_list.append(trj)
            self.input_dict["x"].append(trj.x)
            self.input_dict["y"].append(trj.y)
            self.input_dict["vx"].append(trj.vx)
            self.input_dict["vy"].append(trj.vy)
            self.input_dict["flux"].append(trj.flux)
            self.input_dict["likelihood"].append(trj.lh)
            self.input_dict["obs_count"].append(trj.obs_count)
            self.input_dict["uuid"].append("none")

    def _assert_results_match_dict(self, results, test_dict):
        # Check that the shape of the results are the same.
        self.assertEqual(len(results), len(test_dict["x"]))
        self.assertEqual(set(results.colnames), set(test_dict.keys()))

        for col in results.colnames:
            # Check that all columns match except UUID, which is dynamically assigned.
            if col != "uuid":
                for i in range(len(results)):
                    self.assertEqual(results[col][i], test_dict[col][i])

    def test_empty(self):
        table = Results()
        self.assertEqual(len(table), 0)
        self.assertEqual(len(table.colnames), 8)
        self.assertEqual(table.get_num_times(), 0)
        self.assertIsNone(table.wcs)
        self.assertIsNone(table.mjd_mid)

        self.assertTrue("x" in table.colnames)
        self.assertTrue("y" in table.colnames)
        self.assertTrue("vx" in table.colnames)
        self.assertTrue("vy" in table.colnames)
        self.assertTrue("flux" in table.colnames)
        self.assertTrue("likelihood" in table.colnames)
        self.assertTrue("obs_count" in table.colnames)
        self.assertTrue("uuid" in table.colnames)

        # Check that we don't crash on updating the likelihoods.
        table._update_likelihood()

    def test_from_trajectories(self):
        table = Results.from_trajectories(self.trj_list)
        self.assertEqual(len(table), self.num_entries)
        self.assertEqual(len(table.colnames), 8)
        self.assertIsNone(table.wcs)
        self.assertIsNone(table.mjd_mid)

        self.assertTrue("x" in table.colnames)
        self.assertTrue("y" in table.colnames)
        self.assertTrue("vx" in table.colnames)
        self.assertTrue("vy" in table.colnames)
        self.assertTrue("flux" in table.colnames)
        self.assertTrue("likelihood" in table.colnames)
        self.assertTrue("obs_count" in table.colnames)
        self.assertTrue("uuid" in table.colnames)
        self._assert_results_match_dict(table, self.input_dict)

        # Test that we automatically generate unique ids.
        self.assertEqual(len(np.unique(table["uuid"])), len(table))

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

        # Check that the UUID column matches.
        self.assertTrue(np.array_equal(table1["uuid"], table2["uuid"]))

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

    def test_sort(self):
        trj_list = [
            Trajectory(x=0, y=0, vx=0, vy=0, lh=100.0, obs_count=10),
            Trajectory(x=1, y=1, vx=0, vy=0, lh=110.0, obs_count=9),
            Trajectory(x=2, y=2, vx=0, vy=0, lh=90.0, obs_count=8),
            Trajectory(x=3, y=3, vx=0, vy=0, lh=120.0, obs_count=11),
            Trajectory(x=4, y=4, vx=0, vy=0, lh=80.0, obs_count=15),
            Trajectory(x=5, y=5, vx=0, vy=0, lh=85.0, obs_count=12),
            Trajectory(x=6, y=6, vx=0, vy=0, lh=75.0, obs_count=5),
            Trajectory(x=7, y=7, vx=0, vy=0, lh=125.0, obs_count=14),
        ]
        table = Results.from_trajectories(trj_list)
        self.assertTrue(np.array_equal(table["x"], [0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertTrue(np.array_equal(table["y"], [0, 1, 2, 3, 4, 5, 6, 7]))

        table.sort("likelihood")
        self.assertTrue(np.array_equal(table["x"], [7, 3, 1, 0, 2, 5, 4, 6]))
        self.assertTrue(np.array_equal(table["y"], [7, 3, 1, 0, 2, 5, 4, 6]))

        table.sort("obs_count")
        self.assertTrue(np.array_equal(table["x"], [4, 7, 5, 3, 0, 1, 2, 6]))
        self.assertTrue(np.array_equal(table["y"], [4, 7, 5, 3, 0, 1, 2, 6]))

        table.sort("x", descending=False)
        self.assertTrue(np.array_equal(table["x"], [0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertTrue(np.array_equal(table["y"], [0, 1, 2, 3, 4, 5, 6, 7]))

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

        # Without obs_valid: Check the the data has been inserted and the
        # statistics have been updated.
        table.add_psi_phi_data(psi_array, phi_array)
        for i in range(num_to_use):
            self.assertEqual(len(table["psi_curve"][i]), 4)
            self.assertEqual(len(table["phi_curve"][i]), 4)
            self.assertEqual(table["obs_count"][i], 4)

        # With obs_valid: Check the the data has been inserted and the
        # statistics have been updated.
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
        table.mjd_mid = 59000.0 + np.array([1, 2, 3, 4, 5])

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
                table.write_table(file_path, overwrite=False)

            # We can overwrite with droped columns and additional meta data.
            table.write_table(
                file_path,
                overwrite=True,
                extra_meta={"other": 100.0},
            )

            table3 = Results.read_table(file_path)
            self.assertEqual(len(table2), max_save)

            # We saved the additional meta data, including the WCS.
            self.assertTrue(np.array_equal(table3.table.meta["mjd_mid"], table.mjd_mid))
            self.assertEqual(table3.table.meta["other"], 100.0)
            self.assertIsNotNone(table3.wcs)
            self.assertTrue(wcs_fits_equal(table3.wcs, fake_wcs))

    def test_to_from_table_file_empty(self):
        table = Results()
        self.assertEqual(len(table), 0)

        # Create a fake WCS to use for serialization tests.
        fake_wcs = make_fake_wcs(25.0, -7.5, 800, 600, deg_per_pixel=0.01)
        table.wcs = fake_wcs

        # Add fake times.
        table.mjd_mid = 59000.0 + np.array([1, 2, 3, 4, 5])

        # Test read/write to file.
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "results.ecsv")
            table.write_table(file_path)
            self.assertTrue(Path(file_path).is_file())

            table2 = Results.read_table(file_path)
            self.assertEqual(len(table2), 0)

    def test_to_from_table_file_formats(self):
        max_save = 5
        table = Results.from_trajectories(self.trj_list[0:max_save], track_filtered=True)
        table.table["other"] = [i for i in range(max_save)]
        self.assertEqual(len(table), max_save)

        # Create a fake WCS to use for serialization tests.
        fake_wcs = make_fake_wcs(25.0, -7.5, 800, 600, deg_per_pixel=0.01)
        table.wcs = fake_wcs

        # Add fake times.
        table.mjd_mid = 59000.0 + np.array([1, 2, 3, 4, 5])

        # Test read/write to file.
        with tempfile.TemporaryDirectory() as dir_name:
            for fmt in ["ecsv", "parq", "parquet", "hdf5"]:
                with self.subTest(fmt_used=fmt):
                    file_path = os.path.join(dir_name, f"results.{fmt}")
                    table.write_table(file_path)
                    self.assertTrue(Path(file_path).is_file())

                    table2 = Results.read_table(file_path)
                    self.assertEqual(len(table2), max_save)

                    # Check that we saved the additional meta data, including the WCS.
                    self.assertTrue(np.array_equal(table2.table.meta["mjd_mid"], table.mjd_mid))
                    self.assertIsNotNone(table2.wcs)
                    self.assertTrue(wcs_fits_equal(table2.wcs, fake_wcs))

            # Check that we fail when using an unsupported file type.
            with self.assertRaises(ValueError):
                file_path = os.path.join(dir_name, f"results.fits")
                table.write_table(file_path)

    def test_write_and_load_data_column(self):
        # Create a table with two extra columns one of scalars and one of lists.
        table = Results.from_trajectories(self.trj_list)
        table.table["extra_scalar"] = [100 + i for i in range(self.num_entries)]
        table.table["extra_array"] = [np.array([100 + i, 100 - i, 100 * i]) for i in range(self.num_entries)]

        # Try outputting a single column using the cross product of all the supported
        # formats and the two columns.
        with tempfile.TemporaryDirectory() as dir_name:
            for fmt in ["npy", "ecsv", "parq", "parquet", "fits"]:
                for col in ["extra_scalar", "extra_array"]:
                    with self.subTest(fmt_used=fmt, col_written=col):
                        file_path = os.path.join(dir_name, f"{col}.{fmt}")

                        # Can't load if the file is not there.
                        with self.assertRaises(FileNotFoundError):
                            table.load_column(file_path, col)

                        # Before loading, the column is not in the table.
                        table2 = Results.from_trajectories(self.trj_list)
                        assert col not in table2.colnames

                        # Save the data.
                        table.write_column(col, file_path)

                        # Check that we can read the column in the other table.
                        table2.load_column(file_path, col)
                        self.assertTrue(col in table2.colnames)

    def test_write_and_load_column_np(self):
        table = Results.from_trajectories(self.trj_list)
        self.assertFalse("all_stamps" in table.colnames)

        # Create a table with an extra column.
        table2 = Results.from_trajectories(self.trj_list)
        all_stamps = []
        for i in range(len(table)):
            all_stamps.append([np.full((5, 5), i), np.full((5, 5), i + 10)])
        table2.table["all_stamps"] = all_stamps
        self.assertTrue("all_stamps" in table2.colnames)

        # Try outputting the Results
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

    def test_write_and_load_stamps_column_fits(self):
        # Create a table with an extra column of all stamps with 21 x 21 stamps
        # at 100 times and a coadd column with 51 x 51 stamps.
        table = Results.from_trajectories(self.trj_list)
        table.table["all_stamps"] = [np.zeros((100, 21, 21)) + i / 100.0 for i in range(self.num_entries)]
        table.table["codd_mean"] = [np.zeros((51, 51)) + i / 100.0 for i in range(self.num_entries)]

        # Try outputting the Results
        with tempfile.TemporaryDirectory() as dir_name:
            for col in ["all_stamps", "codd_mean"]:
                with self.subTest(col_written=col):
                    file_path = os.path.join(dir_name, f"{col}.fits")
                    table.write_column(col, file_path)

                    # Load the results into a new data structure and confirm they match.
                    table2 = Results.from_trajectories(self.trj_list)
                    self.assertFalse(col in table2.colnames)

                    table2.load_column(file_path, col)
                    self.assertTrue(col in table.colnames)
                    for i in range(self.num_entries):
                        self.assertTrue(np.allclose(table.table[col][i], table2.table[col][i]))

    def test_read_write_aux_columns(self):
        # Create a table with an extra column of all stamps with 21 x 21 stamps
        # at 100 times and a coadd column with 51 x 51 stamps.
        table = Results.from_trajectories(self.trj_list)
        table.table["all_stamps"] = [np.zeros((100, 21, 21)) + i / 100.0 for i in range(self.num_entries)]
        table.table["coadd_mean"] = [np.zeros((51, 51)) + i / 100.0 for i in range(self.num_entries)]

        # Test read/write to file.
        with tempfile.TemporaryDirectory() as dir_name:
            # Write out the image columns as fits files, removing them from the table.
            file_path_all_stamps = os.path.join(dir_name, f"results_all_stamps.fits")
            table.write_column("all_stamps", file_path_all_stamps)
            table.remove_column("all_stamps")
            self.assertFalse("all_stamps" in table.colnames)
            self.assertTrue(Path(file_path_all_stamps).is_file())

            file_path_coadd_mean = os.path.join(dir_name, f"results_coadd_mean.fits")
            table.write_column("coadd_mean", file_path_coadd_mean)
            table.remove_column("coadd_mean")
            self.assertFalse("coadd_mean" in table.colnames)
            self.assertTrue(Path(file_path_coadd_mean).is_file())

            # Save the main table as a parquet file.
            file_path_main = os.path.join(dir_name, f"results.parquet")
            table.write_table(file_path_main)
            self.assertTrue(Path(file_path_main).is_file())

            # Find and load in any auxiliary columns.
            table2 = Results.read_table(file_path_main, load_aux_files=True)
            self.assertEqual(len(table2), self.num_entries)
            self.assertTrue("all_stamps" in table2.colnames)
            self.assertTrue("coadd_mean" in table2.colnames)

            for i in range(self.num_entries):
                self.assertTrue(np.allclose(table2["all_stamps"][i], np.zeros((100, 21, 21)) + i / 100.0))
                self.assertTrue(np.allclose(table2["coadd_mean"][i], np.zeros((51, 51)) + i / 100.0))

    def test_write_filter_stats(self):
        table = Results.from_trajectories(self.trj_list)
        table.filter_rows([1, 3, 4, 5, 6, 7, 8, 9], label="filter1")
        table.filter_rows([1, 2, 3, 4, 7], label="filter2")

        # Try outputting the Results
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

    def test_write_results_to_files_destructive(self):
        # Create a table with an extra column of all stamps with 21 x 21 stamps
        # at 25 times and a coadd column with 31 x 31 stamps.
        table = Results.from_trajectories(self.trj_list)
        table.table["all_stamps"] = [np.zeros((25, 21, 21)) + i / 50.0 for i in range(self.num_entries)]
        table.table["coadd_mean"] = [np.zeros((31, 31)) + i / 50.0 for i in range(self.num_entries)]
        table.table["psi_curve"] = [np.zeros(10) + i for i in range(self.num_entries)]
        table.table["phi_curve"] = [np.zeros(10) + i for i in range(self.num_entries)]

        # Test writing the results to files.
        with tempfile.TemporaryDirectory() as dir_name:
            main_file_path = Path(dir_name) / "results.parquet"
            write_results_to_files_destructive(
                main_file_path,
                table,
                extra_meta={"test_meta": "value"},
                separate_col_files=["all_stamps", "coadd_mean", "psi_curve"],
                drop_columns=["phi_curve"],
            )
            self.assertTrue(main_file_path.is_file())
            self.assertTrue(Path(dir_name, "results_all_stamps.fits").is_file())
            self.assertTrue(Path(dir_name, "results_coadd_mean.fits").is_file())
            self.assertTrue(Path(dir_name, "results_psi_curve.parquet").is_file())

            # Read the table and confirm that we have the expected columns.
            table2 = Results.read_table(main_file_path, load_aux_files=True)
            self.assertEqual(len(table2), self.num_entries)
            self.assertTrue("all_stamps" in table2.colnames)
            self.assertTrue("coadd_mean" in table2.colnames)
            self.assertTrue("psi_curve" in table2.colnames)
            self.assertFalse("phi_curve" in table2.colnames)

            # Check the metadata in the main file.
            self.assertEqual(table2.table.meta["test_meta"], "value")
            self.assertEqual(table2.table.meta["dropped_columns"], ["phi_curve"])
            self.assertEqual(
                table2.table.meta["separate_col_files"],
                ["all_stamps", "coadd_mean", "psi_curve"],
            )

    def test_read_table_chunks(self):
        """Test the chunked reading of a parquet file."""
        max_save = 15
        table = Results.from_trajectories(
            self.trj_list[0:max_save] if max_save <= self.num_entries else self.trj_list * 2
        )

        # Make sure we have enough entries for meaningful chunking
        while len(table) < max_save:
            table.extend(Results.from_trajectories(self.trj_list))
        table = Results.from_trajectories([self.trj_list[i % self.num_entries] for i in range(max_save)])

        # Add fake times to test metadata extraction
        table.mjd_mid = 59000.0 + np.arange(5)

        # Create a fake WCS
        fake_wcs = make_fake_wcs(25.0, -7.5, 800, 600, deg_per_pixel=0.01)
        table.wcs = fake_wcs

        # Test with parquet format only (chunked reading requires parquet)
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "results.parquet")
            table.write_table(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Test 1: Read in chunks and verify all data is recovered
            chunk_size = 5
            all_chunks = list(Results.read_table_chunks(file_path, chunk_size=chunk_size))

            # Should have 3 chunks (15 rows / 5 per chunk)
            self.assertEqual(len(all_chunks), 3)

            # Each chunk should have the correct number of rows
            self.assertEqual(len(all_chunks[0]), 5)
            self.assertEqual(len(all_chunks[1]), 5)
            self.assertEqual(len(all_chunks[2]), 5)

            # Each chunk should have mjd_mid attached
            for chunk in all_chunks:
                self.assertIsNotNone(chunk.mjd_mid)
                self.assertEqual(len(chunk.mjd_mid), 5)
                self.assertTrue(np.array_equal(chunk.mjd_mid, table.mjd_mid))

            # Verify that concatenating chunks gives us the original data
            total_rows = sum(len(c) for c in all_chunks)
            self.assertEqual(total_rows, max_save)

            # Test 2: Chunk size larger than file should return single chunk
            all_chunks = list(Results.read_table_chunks(file_path, chunk_size=1000))
            self.assertEqual(len(all_chunks), 1)
            self.assertEqual(len(all_chunks[0]), max_save)

            # Test 3: Verify WCS is attached to chunks
            all_chunks = list(Results.read_table_chunks(file_path, chunk_size=5))
            for chunk in all_chunks:
                self.assertIsNotNone(chunk.wcs)
                self.assertTrue(wcs_fits_equal(chunk.wcs, fake_wcs))

    def test_read_table_chunks_file_not_found(self):
        """Test that read_table_chunks raises error for missing file."""
        with self.assertRaises(FileNotFoundError):
            list(Results.read_table_chunks("/nonexistent/file.parquet"))

    def test_read_table_chunks_unsupported_format(self):
        """Test that read_table_chunks raises error for non-parquet files."""
        with tempfile.TemporaryDirectory() as dir_name:
            # Create an ecsv file
            table = Results.from_trajectories(self.trj_list)
            file_path = os.path.join(dir_name, "results.ecsv")
            table.write_table(file_path)

            # Should raise error for non-parquet
            with self.assertRaises(ValueError):
                list(Results.read_table_chunks(file_path))

    def test_parquet_image_column_roundtrip(self):
        """Test that 2D image columns survive parquet serialization via metadata-based reshaping."""
        table = Results.from_trajectories(self.trj_list)

        # Add 2D image columns (coadds) - these will be flattened by parquet
        coadd_shape = (31, 31)
        table.table["coadd_mean"] = [np.full(coadd_shape, i / 10.0) for i in range(self.num_entries)]
        table.table["coadd_median"] = [np.full(coadd_shape, i / 20.0) for i in range(self.num_entries)]

        # Add a 1D curve column - should stay 1D
        table.table["psi_curve"] = [np.arange(25) + i for i in range(self.num_entries)]

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "results.parquet")

            # Write with explicit image_columns to ensure metadata is stored
            table.write_table(file_path, image_columns=["coadd_mean", "coadd_median"])
            self.assertTrue(Path(file_path).is_file())

            # Read back and verify shapes are restored
            table2 = Results.read_table(file_path)

            # Check that image columns have correct 2D shape
            self.assertEqual(table2["coadd_mean"][0].shape, coadd_shape)
            self.assertEqual(table2["coadd_median"][0].shape, coadd_shape)

            # Check that values are preserved
            for i in range(self.num_entries):
                self.assertTrue(np.allclose(table2["coadd_mean"][i], np.full(coadd_shape, i / 10.0)))
                self.assertTrue(np.allclose(table2["coadd_median"][i], np.full(coadd_shape, i / 20.0)))

            # Check that 1D curve column is still 1D
            self.assertEqual(table2["psi_curve"][0].shape, (25,))

    def test_parquet_image_column_roundtrip_chunks(self):
        """Test that 2D image columns survive chunked parquet reading."""
        table = Results.from_trajectories(self.trj_list)

        # Add 2D image columns
        coadd_shape = (21, 21)
        table.table["coadd_mean"] = [np.full(coadd_shape, i / 10.0) for i in range(self.num_entries)]

        # Add metadata (WCS, mjd_mid) to test full metadata extraction
        table.mjd_mid = 59000.0 + np.arange(5)
        fake_wcs = make_fake_wcs(25.0, -7.5, 800, 600, deg_per_pixel=0.01)
        table.wcs = fake_wcs

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "results.parquet")
            table.write_table(file_path, image_columns=["coadd_mean"])

            # Read in chunks
            chunks = list(Results.read_table_chunks(file_path, chunk_size=4))

            # Should have 3 chunks (10 rows / 4 per chunk = 2 full + 1 partial)
            self.assertEqual(len(chunks), 3)

            # Each chunk should have correctly shaped coadds
            row_idx = 0
            for chunk in chunks:
                for i in range(len(chunk)):
                    self.assertEqual(chunk["coadd_mean"][i].shape, coadd_shape)
                    expected_val = row_idx / 10.0
                    self.assertTrue(np.allclose(chunk["coadd_mean"][i], np.full(coadd_shape, expected_val)))
                    row_idx += 1

                # Each chunk should have WCS and mjd_mid
                self.assertIsNotNone(chunk.wcs)
                self.assertIsNotNone(chunk.mjd_mid)

    def test_is_image_like_with_metadata(self):
        """Test that is_image_like uses metadata when available."""
        table = Results.from_trajectories(self.trj_list)

        # Add a 1D array column that we'll mark as an image via metadata
        table.table["fake_coadd"] = [np.zeros(100) for _ in range(self.num_entries)]

        # Without metadata, 1D array should NOT be image-like
        self.assertFalse(table.is_image_like("fake_coadd"))

        # Set metadata marking it as an image column
        table.table.meta["image_columns"] = ["fake_coadd"]

        # Now it should be considered image-like due to metadata
        self.assertTrue(table.is_image_like("fake_coadd"))

        # A column not in metadata and not 2D+ should not be image-like
        table.table["some_1d_data"] = [np.zeros(50) for _ in range(self.num_entries)]
        self.assertFalse(table.is_image_like("some_1d_data"))

    def test_write_results_destructive_explicit_image_columns(self):
        """Test write_results_to_files_destructive with explicit image_columns parameter."""
        table = Results.from_trajectories(self.trj_list)

        # Add columns - all are numpy arrays but only some are images
        table.table["coadd_mean"] = [np.zeros((21, 21)) + i for i in range(self.num_entries)]
        table.table["psi_curve"] = [np.zeros(25) + i for i in range(self.num_entries)]
        table.table["phi_curve"] = [np.zeros(25) + i for i in range(self.num_entries)]

        with tempfile.TemporaryDirectory() as dir_name:
            main_file_path = Path(dir_name) / "results.parquet"

            # Use explicit image_columns - only coadd_mean is an image
            write_results_to_files_destructive(
                main_file_path,
                table,
                separate_col_files=["coadd_mean", "psi_curve"],
                image_columns=["coadd_mean"],  # Explicitly mark only coadd_mean as image
            )

            # coadd_mean should be saved as FITS (it's an image)
            self.assertTrue(Path(dir_name, "results_coadd_mean.fits").is_file())

            # psi_curve should be saved as parquet (it's NOT an image)
            self.assertTrue(Path(dir_name, "results_psi_curve.parquet").is_file())

    def test_write_column_explicit_is_image(self):
        """Test write_column with explicit is_image parameter."""
        table = Results.from_trajectories(self.trj_list)

        # Add a 2D array that would normally be detected as an image
        table.table["data_2d"] = [np.zeros((10, 10)) + i for i in range(self.num_entries)]

        with tempfile.TemporaryDirectory() as dir_name:
            # Force it to be written as non-image (BinTable) even though it's 2D
            file_path = os.path.join(dir_name, "data_2d_as_table.fits")
            table.write_column("data_2d", file_path, is_image=False)

            # Load it back and verify
            table2 = Results.from_trajectories(self.trj_list)
            table2.load_column(file_path, "data_2d")
            self.assertTrue("data_2d" in table2.colnames)

            # Now write it as an image (the default for 2D data)
            file_path2 = os.path.join(dir_name, "data_2d_as_image.fits")
            table.write_column("data_2d", file_path2, is_image=True)

            # Both should be loadable
            table3 = Results.from_trajectories(self.trj_list)
            table3.load_column(file_path2, "data_2d")
            self.assertTrue("data_2d" in table3.colnames)

    def test_parse_table_metadata_empty(self):
        """Test _parse_table_metadata with empty and partial metadata."""
        # Empty dict should return all None
        wcs, mjd_mid, shapes = Results._parse_table_metadata({})
        self.assertIsNone(wcs)
        self.assertIsNone(mjd_mid)
        self.assertIsNone(shapes)

        # None should return all None
        wcs, mjd_mid, shapes = Results._parse_table_metadata(None)
        self.assertIsNone(wcs)
        self.assertIsNone(mjd_mid)
        self.assertIsNone(shapes)

        # Partial metadata - only mjd_utc_mid
        meta = {"mjd_utc_mid": [59000.0, 59001.0]}
        wcs, mjd_mid, shapes = Results._parse_table_metadata(meta)
        self.assertIsNone(wcs)
        self.assertEqual(mjd_mid, [59000.0, 59001.0])
        self.assertIsNone(shapes)

        # Partial metadata - only image_column_shapes
        meta = {"image_column_shapes": {"coadd": [21, 21]}}
        wcs, mjd_mid, shapes = Results._parse_table_metadata(meta)
        self.assertIsNone(wcs)
        self.assertIsNone(mjd_mid)
        self.assertEqual(shapes, {"coadd": [21, 21]})

    def test_reshape_image_columns_edge_cases(self):
        """Test _reshape_image_columns with edge cases."""
        # Test with empty results - should not crash
        empty_table = Results()
        empty_table._reshape_image_columns({"coadd": (21, 21)})
        self.assertEqual(len(empty_table), 0)

        # Test with None shapes - should do nothing
        table = Results.from_trajectories(self.trj_list)
        table.table["data"] = [np.zeros(100) for _ in range(self.num_entries)]
        original_shape = table["data"][0].shape
        table._reshape_image_columns(None)
        self.assertEqual(table["data"][0].shape, original_shape)

        # Test with empty shapes dict - should do nothing
        table._reshape_image_columns({})
        self.assertEqual(table["data"][0].shape, original_shape)

        # Test with column not in table - should not crash
        table._reshape_image_columns({"nonexistent_column": (10, 10)})

        # Test with incompatible shape - should warn but not crash
        # 100 elements can't be reshaped to (7, 7) = 49 elements
        table._reshape_image_columns({"data": (7, 7)})
        # Shape should remain unchanged due to error
        self.assertEqual(table["data"][0].shape, original_shape)

    def test_detect_image_columns(self):
        """Test the _detect_image_columns helper method."""
        table = Results.from_trajectories(self.trj_list)

        # Add various column types
        table.table["coadd_2d"] = [np.zeros((21, 21)) for _ in range(self.num_entries)]
        table.table["stamps_3d"] = [np.zeros((10, 21, 21)) for _ in range(self.num_entries)]
        table.table["curve_1d"] = [np.zeros(25) for _ in range(self.num_entries)]

        # Auto-detect (no explicit list)
        shapes = table._detect_image_columns(None)
        self.assertIn("coadd_2d", shapes)
        self.assertIn("stamps_3d", shapes)
        self.assertNotIn("curve_1d", shapes)  # 1D should not be detected
        self.assertEqual(shapes["coadd_2d"], (21, 21))
        self.assertEqual(shapes["stamps_3d"], (10, 21, 21))

        # Explicit list
        shapes = table._detect_image_columns(["coadd_2d"])
        self.assertIn("coadd_2d", shapes)
        self.assertNotIn("stamps_3d", shapes)
        self.assertEqual(shapes["coadd_2d"], (21, 21))

        # Explicit list with column that doesn't exist - should skip it
        shapes = table._detect_image_columns(["coadd_2d", "nonexistent"])
        self.assertIn("coadd_2d", shapes)
        self.assertNotIn("nonexistent", shapes)


if __name__ == "__main__":
    unittest.main()
