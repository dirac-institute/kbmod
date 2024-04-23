import copy
import numpy as np
import os
import numpy as np
import tempfile
import unittest

from astropy.table import Table
from astropy.wcs import WCS
from pathlib import Path

from kbmod.file_utils import *
from kbmod.result_list import *
from kbmod.search import *


class test_result_data_row(unittest.TestCase):
    def setUp(self):
        self.trj = Trajectory()
        self.trj.obs_count = 4

        self.times = [1.0, 2.0, 3.0, 4.0]
        self.num_times = len(self.times)
        self.rdr = ResultRow(self.trj, self.num_times)
        self.rdr.set_psi_phi([1.0, 1.1, 1.2, 1.3], [1.0, 1.0, 0.0, 2.0])

        example_stamp = np.ones((5, 5))
        self.rdr.all_stamps = np.array([np.copy(example_stamp) for _ in range(4)])

    def test_get_boolean_valid_indices(self):
        self.assertEqual(self.rdr.valid_indices_as_booleans(), [True, True, True, True])

        self.rdr.filter_indices([1, 2])
        self.assertEqual(self.rdr.valid_indices_as_booleans(), [False, True, True, False])

    def test_equal(self):
        row_copy = copy.deepcopy(self.rdr)
        self.assertTrue(self.rdr == row_copy)

        # Change something in the trajectory
        row_copy.trajectory.x = 20
        self.assertFalse(self.rdr == row_copy)
        row_copy.trajectory.x = self.rdr.trajectory.x
        self.assertTrue(self.rdr == row_copy)

        # Change a value in the psi array
        row_copy.psi_curve[2] = 1.9
        self.assertFalse(self.rdr == row_copy)
        row_copy.psi_curve[2] = self.rdr.psi_curve[2]
        self.assertTrue(self.rdr == row_copy)

        # None out all all stamps
        row_copy.all_stamps = None
        self.assertFalse(self.rdr == row_copy)

    def test_filter(self):
        self.assertEqual(self.rdr.valid_indices, [0, 1, 2, 3])
        self.assertTrue(np.allclose(self.rdr.valid_times(self.times), [1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(self.rdr.trajectory.obs_count, 4)
        self.assertAlmostEqual(self.rdr.trajectory.flux, 1.15)
        self.assertAlmostEqual(self.rdr.trajectory.lh, 2.3)

        self.rdr.filter_indices([0, 2, 3])
        self.assertEqual(self.rdr.valid_indices, [0, 2, 3])
        self.assertEqual(self.rdr.valid_times(self.times), [1.0, 3.0, 4.0])

        # The values within the Trajectory object *should* change.
        self.assertEqual(self.rdr.trajectory.obs_count, 3)
        self.assertAlmostEqual(self.rdr.trajectory.flux, 1.1666667, delta=1e-5)
        self.assertAlmostEqual(self.rdr.trajectory.lh, 2.020725, delta=1e-5)

        # The curves and stamps should not change.
        self.assertTrue(np.allclose(self.rdr.psi_curve, [1.0, 1.1, 1.2, 1.3]))
        self.assertTrue(np.allclose(self.rdr.phi_curve, [1.0, 1.0, 0.0, 2.0]))
        self.assertEqual(self.rdr.all_stamps.shape, (4, 5, 5))

    def test_set_psi_phi(self):
        self.rdr.set_psi_phi([1.5, 1.1, 1.2, 1.0], [1.0, 0.0, 0.0, 0.5])
        self.assertTrue(np.allclose(self.rdr.psi_curve, [1.5, 1.1, 1.2, 1.0]))
        self.assertTrue(np.allclose(self.rdr.phi_curve, [1.0, 0.0, 0.0, 0.5]))
        self.assertTrue(np.allclose(self.rdr.light_curve, [1.5, 0.0, 0.0, 2.0]))

    def test_compute_likelihood_curve(self):
        self.rdr.set_psi_phi([1.5, 1.1, 1.2, 1.1], [1.0, 0.0, 4.0, 0.25])
        lh = self.rdr.likelihood_curve
        self.assertTrue(np.allclose(lh, [1.5, 0.0, 0.6, 2.2], atol=1e-5))

    def test_to_from_yaml(self):
        yaml_str = self.rdr.to_yaml()
        self.assertGreater(len(yaml_str), 0)

        row2 = ResultRow.from_yaml(yaml_str)
        self.assertAlmostEqual(row2.final_likelihood, 2.3)
        self.assertEqual(row2.valid_indices, [0, 1, 2, 3])
        self.assertEqual(row2.valid_times(self.times), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(row2.trajectory.obs_count, 4)
        self.assertIsNone(row2.stamp)
        self.assertIsNone(row2.pred_ra)
        self.assertIsNone(row2.pred_dec)
        self.assertIsNotNone(row2.all_stamps)
        self.assertEqual(row2.all_stamps.shape[0], 4)
        self.assertEqual(row2.all_stamps.shape[1], 5)
        self.assertEqual(row2.all_stamps.shape[2], 5)
        self.assertEqual(self.rdr, row2)

        self.assertIsNotNone(row2.trajectory)
        self.assertAlmostEqual(row2.trajectory.flux, 1.15)
        self.assertAlmostEqual(row2.trajectory.lh, 2.3)

    def test_from_table_row(self):
        test_dict = {
            "trajectory_x": [],
            "trajectory_y": [],
            "trajectory_vx": [],
            "trajectory_vy": [],
            "obs_count": [],
            "flux": [],
            "likelihood": [],
            "stamp": [],
            "all_stamps": [],
            "valid_indices": [],
            "psi_curve": [],
            "phi_curve": [],
            "pred_ra": [],
            "pred_dec": [],
        }
        self.rdr.append_to_dict(test_dict, expand_trajectory=True)

        trjB = Trajectory(0, 1, 2.0, -3.0, 10.0, 21.0, 3)
        rowB = ResultRow(trjB, 4)
        rowB.append_to_dict(test_dict, expand_trajectory=True)

        # Test that we can extract them from a row.
        data = Table(test_dict)
        self.assertEqual(self.rdr, ResultRow.from_table_row(data[0], 4))
        self.assertEqual(rowB, ResultRow.from_table_row(data[1], 4))

        # We fail if no number of times is given.  Unless the table has an
        # appropriate column with that information.
        with self.assertRaises(KeyError):
            _ = ResultRow.from_table_row(data[0])

        test_dict["all_times"] = [self.times, self.times]
        data = Table(test_dict)
        self.assertEqual(self.rdr, ResultRow.from_table_row(data[0]))

        # Test that we can still extract the data without the stamp or all_stamps columns
        del test_dict["stamp"]
        del test_dict["all_stamps"]
        self.assertIsNotNone(ResultRow.from_table_row(data[0], 4))

    def test_compute_predicted_skypos(self):
        self.assertIsNone(self.rdr.pred_ra)
        self.assertIsNone(self.rdr.pred_dec)

        # Fill out the trajectory details
        self.rdr.trajectory.x = 9
        self.rdr.trajectory.y = 9
        self.rdr.trajectory.vx = -1.0
        self.rdr.trajectory.vy = 3.0

        # Create a fake WCS with a known pointing.
        my_wcs = WCS(naxis=2)
        my_wcs.wcs.crpix = [10.0, 10.0]  # Reference point on the image (1-indexed)
        my_wcs.wcs.crval = [45.0, -15.0]  # Reference pointing on the sky
        my_wcs.wcs.cdelt = [0.05, 0.15]  # Pixel step size
        my_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

        times = [0.0, 1.0, 2.0, 3.0]
        self.rdr.compute_predicted_skypos(times, my_wcs)
        self.assertEqual(len(self.rdr.pred_ra), 4)
        self.assertEqual(len(self.rdr.pred_dec), 4)
        self.assertAlmostEqual(self.rdr.pred_ra[0], 45.0, delta=0.01)
        self.assertAlmostEqual(self.rdr.pred_dec[0], -15.0, delta=0.01)


class test_result_list(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

        # Create a fake WCS with a known pointing to use for the (RA, dec) predictions.
        self.my_wcs = WCS(naxis=2)
        self.my_wcs.wcs.crpix = [50.0, 50.0]  # Reference point on the image (1-indexed)
        self.my_wcs.wcs.crval = [45.0, -15.0]  # Reference pointing on the sky
        self.my_wcs.wcs.cdelt = [0.05, 0.05]  # Pixel step size
        self.my_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

    def test_append_single(self):
        rs = ResultList(self.times)
        self.assertEqual(rs.num_results(), 0)

        for i in range(5):
            t = Trajectory()
            t.lh = float(i)
            rs.append_result(ResultRow(t, self.num_times))
        self.assertEqual(rs.num_results(), 5)

        for i in range(5):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].final_likelihood, float(i))

    def test_extend(self):
        rs1 = ResultList(self.times)
        rs2 = ResultList(self.times)
        self.assertEqual(rs1.num_results(), 0)
        self.assertEqual(rs2.num_results(), 0)

        # Fill the first ResultList with 5 rows.
        for i in range(5):
            t = Trajectory()
            t.lh = float(i)
            rs1.append_result(ResultRow(t, self.num_times))

        # Fill a second Result set with 5 different rows.
        for i in range(5):
            t = Trajectory()
            t.lh = float(i) + 5.0
            rs2.append_result(ResultRow(t, self.num_times))

        # Check that each result set has 5 results.
        self.assertEqual(rs1.num_results(), 5)
        self.assertEqual(rs2.num_results(), 5)

        # Append the two results set and check the 10 combined results.
        rs1.extend(rs2)
        self.assertEqual(rs1.num_results(), 10)
        self.assertEqual(rs2.num_results(), 5)
        for i in range(10):
            self.assertIsNotNone(rs1.results[i].trajectory)
            self.assertEqual(rs1.results[i].final_likelihood, float(i))

    def test_clear(self):
        rs = ResultList(self.times)
        for i in range(3):
            t = Trajectory()
            rs.append_result(ResultRow(t, self.num_times))
        self.assertEqual(rs.num_results(), 3)

        rs.clear()
        self.assertEqual(rs.num_results(), 0)

    def test_sort(self):
        rs = ResultList(self.times)
        rs.append_result(ResultRow(Trajectory(x=0, lh=1.0, obs_count=1), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=1, lh=-1.0, obs_count=2), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=2, lh=5.0, obs_count=3), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=3, lh=4.0, obs_count=5), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=4, lh=6.0, obs_count=4), self.num_times))

        # Sort by final likelihood.
        rs.sort()
        self.assertEqual(rs.num_results(), 5)
        expected_order = [4, 2, 3, 0, 1]
        for i, val in enumerate(expected_order):
            self.assertEqual(rs.results[i].trajectory.x, val)

        # Sort by the number of observations.
        rs.sort(key="obs_count", reverse=False)
        self.assertEqual(rs.num_results(), 5)
        expected_order = [0, 1, 2, 4, 3]
        for i, val in enumerate(expected_order):
            self.assertEqual(rs.results[i].trajectory.x, val)

    def test_get_result_values(self):
        rs = ResultList(self.times)
        rs.append_result(ResultRow(Trajectory(x=0, lh=1.0, obs_count=1), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=1, lh=-1.0, obs_count=2), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=2, lh=5.0, obs_count=3), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=3, lh=4.0, obs_count=5), self.num_times))
        rs.append_result(ResultRow(Trajectory(x=4, lh=6.0, obs_count=4), self.num_times))

        # Test getting a list of trajectories.
        trjs = rs.get_result_values("trajectory")
        self.assertEqual(len(trjs), 5)
        for i in range(5):
            self.assertTrue(type(trjs[i]) is Trajectory)

        # Stamps should all be None
        stamps = rs.get_result_values("stamp")
        self.assertEqual(len(stamps), 5)
        for i in range(5):
            self.assertTrue(stamps[i] is None)

        # We can extract sub-attributes
        x_vals = rs.get_result_values("trajectory.x")
        self.assertEqual(len(x_vals), 5)
        for i in range(5):
            self.assertEqual(x_vals[i], i)

        vx_vals = rs.get_result_values("trajectory.vx")
        self.assertEqual(len(vx_vals), 5)
        for i in range(5):
            self.assertEqual(vx_vals[i], 0.0)

        # We get an error if we try to extract an attribute that doesn't exist.
        self.assertRaises(AttributeError, rs.get_result_values, "")
        self.assertRaises(AttributeError, rs.get_result_values, "Not There")
        self.assertRaises(AttributeError, rs.get_result_values, "trajectory.z")

    def test_filter(self):
        rs = ResultList(self.times)
        for i in range(10):
            t = Trajectory()
            t.lh = float(i)
            rs.append_result(ResultRow(t, self.num_times))
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7]
        rs.filter_results(inds)
        self.assertEqual(rs.num_results(), len(inds))
        for i in range(len(inds)):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].final_likelihood, float(inds[i]))

        # Without tracking there should be nothing stored in the ResultList's
        # filtered dictionary.
        self.assertEqual(len(rs.filtered), 0)
        with self.assertRaises(ValueError):
            rs.get_filtered()

        # Without tracking we cannot revert anything.
        with self.assertRaises(ValueError):
            rs.revert_filter()

    def test_filter_dups(self):
        rs = ResultList(self.times, track_filtered=False)
        for i in range(10):
            t = Trajectory()
            t.lh = float(i)
            rs.append_result(ResultRow(t, self.num_times))
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7, 7, 0, 7]
        rs.filter_results(inds)
        self.assertEqual(rs.num_results(), 4)
        self.assertEqual(rs.results[0].final_likelihood, 0.0)
        self.assertEqual(rs.results[1].final_likelihood, 2.0)
        self.assertEqual(rs.results[2].final_likelihood, 6.0)
        self.assertEqual(rs.results[3].final_likelihood, 7.0)

        # Without tracking there should be nothing stored in the ResultList's
        # filtered dictionary.
        self.assertEqual(len(rs.filtered), 0)

    def test_filter_track(self):
        rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            trj = Trajectory(x=i)
            rs.append_result(ResultRow(trj, self.num_times))
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering. First remove elements 0 and 2. Then remove elements
        # 0, 5, and 6 from the resulting list (1, 7, 8 in the original list).
        rs.filter_results([1, 3, 4, 5, 6, 7, 8, 9], label="1")
        self.assertEqual(rs.num_results(), 8)
        rs.filter_results([1, 2, 3, 4, 7], label="2")
        self.assertEqual(rs.num_results(), 5)
        self.assertEqual(rs.results[0].trajectory.x, 3)
        self.assertEqual(rs.results[1].trajectory.x, 4)
        self.assertEqual(rs.results[2].trajectory.x, 5)
        self.assertEqual(rs.results[3].trajectory.x, 6)
        self.assertEqual(rs.results[4].trajectory.x, 9)

        # Check that we can get the correct filtered rows.
        f1 = rs.get_filtered("1")
        self.assertEqual(len(f1), 2)
        self.assertEqual(f1[0].trajectory.x, 0)
        self.assertEqual(f1[1].trajectory.x, 2)

        f2 = rs.get_filtered("2")
        self.assertEqual(len(f2), 3)
        self.assertEqual(f2[0].trajectory.x, 1)
        self.assertEqual(f2[1].trajectory.x, 7)
        self.assertEqual(f2[2].trajectory.x, 8)

        # Check that not passing a label gives us all filtered results.
        f_all = rs.get_filtered()
        self.assertEqual(len(f_all), 5)

    def test_revert_filter(self):
        rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            trj = Trajectory(x=i)
            rs.append_result(ResultRow(trj, self.num_times))
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering. First remove elements 0 and 2. Then remove elements
        # 0, 5, and 6 from the resulting list (1, 7, 8 in the original list). Then
        # remove item 5 (9 from the original list).
        rs.filter_results([1, 3, 4, 5, 6, 7, 8, 9], label="1")
        self.assertEqual(rs.num_results(), 8)
        rs.filter_results([1, 2, 3, 4, 7], label="2")
        self.assertEqual(rs.num_results(), 5)
        rs.filter_results([0, 1, 2, 3], label="3")
        self.assertEqual(rs.num_results(), 4)

        # Test that we can recover the items filtered in stage 1. These are added to
        # end, so we should get [3, 4, 5, 6, 0, 2]
        rs.revert_filter(label="1")
        self.assertEqual(rs.num_results(), 6)
        expected_order = [3, 4, 5, 6, 0, 2]
        for i, value in enumerate(expected_order):
            self.assertEqual(rs.results[i].trajectory.x, value)

        # Test that we can recover the all items if we don't provide a label.
        rs.revert_filter()
        self.assertEqual(rs.num_results(), 10)
        expected_order = [3, 4, 5, 6, 0, 2, 1, 7, 8, 9]
        for i, value in enumerate(expected_order):
            self.assertEqual(rs.results[i].trajectory.x, value)

        with self.assertRaises(KeyError):
            rs.revert_filter(label="wrong")

    def test_compute_predicted_skypos(self):
        rs = ResultList(self.times, track_filtered=True)
        for i in range(5):
            trj = Trajectory(x=49 + i, y=49 + i, vx=2 * i, vy=-3 * i, obs_count=self.num_times - i)

        # Check that we have computed a position for each row and time.
        rs.compute_predicted_skypos(self.my_wcs)
        for row in rs.results:
            self.assertEqual(len(row.pred_ra), len(self.times))
            self.assertEqual(len(row.pred_dec), len(self.times))

    def test_to_from_yaml(self):
        rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            row = ResultRow(Trajectory(), self.num_times)
            row.set_psi_phi(np.array([i] * self.num_times), np.array([0.01 * i] * self.num_times))
            rs.append_result(row)
        rs.compute_predicted_skypos(self.my_wcs)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7]
        rs.filter_results(inds, "test")
        self.assertEqual(rs.num_results(), len(inds))

        # Serialize only the unfiltered results.
        yaml_str_a = rs.to_yaml()
        self.assertGreater(len(yaml_str_a), 0)

        rs_a = ResultList.from_yaml(yaml_str_a)
        self.assertEqual(len(rs_a.results), len(inds))
        for i in range(len(inds)):
            self.assertAlmostEqual(rs_a.results[i].psi_curve[0], inds[i])
            self.assertAlmostEqual(rs_a.results[i].phi_curve[0], 0.01 * inds[i])
            self.assertEqual(len(rs_a.results[i].pred_ra), self.num_times)
            self.assertEqual(len(rs_a.results[i].pred_dec), self.num_times)
        self.assertFalse(rs_a.track_filtered)
        self.assertEqual(len(rs_a.filtered), 0)

        # Serialize the filtered results as well
        yaml_str_b = rs.to_yaml(serialize_filtered=True)
        self.assertGreater(len(yaml_str_b), 0)

        rs_b = ResultList.from_yaml(yaml_str_b)
        self.assertEqual(len(rs_b.results), len(inds))
        for i in range(len(inds)):
            self.assertAlmostEqual(rs_b.results[i].psi_curve[0], inds[i])
            self.assertAlmostEqual(rs_b.results[i].phi_curve[0], 0.01 * inds[i])
        self.assertTrue(rs_b.track_filtered)
        self.assertEqual(len(rs_b.filtered), 1)
        self.assertEqual(len(rs_b.filtered["test"]), 10 - len(inds))

    def test_to_from_table(self):
        """Check that we correctly dump the data to a astropy Table"""
        rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            # Flux and likelihood will be auto calculated during set_psi_phi()
            trj = Trajectory(x=i, y=2 * i, vx=100.0 - i, vy=-i, obs_count=self.num_times - i)
            row = ResultRow(trj, self.num_times)
            row.set_psi_phi(np.array([i] * self.num_times), np.array([0.01 * i] * self.num_times))
            row.stamp = np.ones((10, 10))
            row.all_stamps = np.array([np.ones((10, 10)) for _ in range(self.num_times)])
            rs.append_result(row)
        rs.compute_predicted_skypos(self.my_wcs)

        table = rs.to_table()
        self.assertEqual(len(table), 10)
        for i in range(10):
            self.assertEqual(table["trajectory_x"][i], i)
            self.assertEqual(table["trajectory_y"][i], 2 * i)
            self.assertEqual(table["trajectory_vx"][i], 100.0 - i)
            self.assertEqual(table["trajectory_vy"][i], -i)
            self.assertEqual(table["obs_count"][i], self.num_times - i)
            self.assertAlmostEqual(table["flux"][i], rs.results[i].trajectory.flux, delta=1e-5)
            self.assertAlmostEqual(table["likelihood"][i], rs.results[i].trajectory.lh, delta=1e-5)
            self.assertEqual(table["stamp"][i].shape, (10, 10))
            self.assertEqual(len(table["all_stamps"][i]), self.num_times)
            self.assertEqual(len(table["valid_indices"][i]), self.num_times)
            self.assertEqual(len(table["psi_curve"][i]), self.num_times)
            self.assertEqual(len(table["phi_curve"][i]), self.num_times)
            self.assertEqual(len(table["pred_ra"][i]), self.num_times)
            self.assertEqual(len(table["pred_dec"][i]), self.num_times)
            self.assertEqual(table["index"][i], i)

            for j in range(self.num_times):
                self.assertEqual(table["all_stamps"][i][j].shape, (10, 10))
                self.assertEqual(table["valid_indices"][i][j], j)
                self.assertEqual(table["psi_curve"][i][j], i)
                self.assertEqual(table["phi_curve"][i][j], 0.01 * i)

        # Check that we can extract from the table
        rs2 = ResultList.from_table(table, self.times, track_filtered=True)
        self.assertEqual(rs, rs2)

        # We cannot reconstruct without a list of times.
        with self.assertRaises(KeyError):
            _ = ResultList.from_table(table)

        # Filter the result list.
        inds = [1, 2, 5, 6, 7, 8, 9]
        rs.filter_results(inds, "test")
        self.assertEqual(rs.num_results(), len(inds))

        # Check that we can extract the unfiltered table.
        table2 = rs.to_table()
        self.assertEqual(len(table2), len(inds))
        for i in range(len(inds)):
            self.assertEqual(table2["trajectory_x"][i], inds[i])

        # Check that we can extract the filtered entries.
        table3 = rs.to_table(filtered_label="test")
        self.assertEqual(len(table3), 10 - len(inds))

        # Check that we get an error if the filtered label does not exist.
        with self.assertRaises(KeyError):
            rs.to_table(filtered_label="test2")

    def test_sync_table_indices(self):
        """Check that we correctly sync the table data with an existing ResultList"""
        rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            trj = Trajectory(x=i, y=2 * i, vx=100.0 - i, vy=-i, obs_count=self.num_times - i)
            row = ResultRow(trj, self.num_times)
            rs.append_result(row)
        table = rs.to_table()
        self.assertEqual(len(table), 10)

        # Filter the table to specific indices.
        inds_to_keep = [0, 1, 3, 7, 9]
        table = table[inds_to_keep]
        self.assertEqual(len(table), len(inds_to_keep))

        # Sync with the ResultList and confirm both are updated.
        rs.sync_table_indices(table)
        self.assertEqual(len(rs), len(inds_to_keep))
        for i, row in enumerate(rs.results):
            self.assertEqual(row.trajectory.x, inds_to_keep[i])
            self.assertEqual(table["trajectory_x"][i], inds_to_keep[i])
            self.assertEqual(table["index"][i], i)

    def test_to_from_table_file(self):
        rs = ResultList(self.times, track_filtered=False)
        for i in range(10):
            # Flux and likelihood will be auto calculated during set_psi_phi()
            trj = Trajectory(x=i, y=2 * i, vx=100.0 - i, vy=-i, obs_count=self.num_times - i)
            row = ResultRow(trj, self.num_times)
            row.set_psi_phi(np.array([i] * self.num_times), np.array([0.01 * i] * self.num_times))
            row.stamp = np.ones((10, 10))
            row.all_stamps = None
            rs.append_result(row)

        # Test read/write to file.
        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "results.ecsv")
            self.assertFalse(Path(file_path).is_file())

            rs.write_table(file_path)
            self.assertTrue(Path(file_path).is_file())

            rs2 = ResultList.read_table(file_path)
            self.assertEqual(rs, rs2)

    def test_save_results(self):
        """Test the legacy save into a bunch of individual files."""
        times = [0.0, 1.0, 2.0]

        # Fill the ResultList with 3 fake rows.
        rs = ResultList(times)
        for i in range(3):
            row = ResultRow(Trajectory(), 3)
            row.set_psi_phi([0.1, 0.2, 0.3], [1.0, 1.0, 0.5])
            row.filter_indices([t for t in range(i + 1)])
            rs.append_result(row)

        # Try outputting the ResultList
        with tempfile.TemporaryDirectory() as dir_name:
            rs.save_to_files(dir_name, "tmp")

            # Check the results_ file.
            fname = os.path.join(dir_name, "results_tmp.txt")
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_results_file_as_trajectories(fname)
            self.assertEqual(len(data), 3)

            # Check the psi_ file.
            fname = os.path.join(dir_name, "psi_tmp.txt")
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d.tolist(), [0.1, 0.2, 0.3])

            # Check the phi_ file.
            fname = os.path.join(dir_name, "phi_tmp.txt")
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d.tolist(), [1.0, 1.0, 0.5])

            # Check the lc_ file.
            fname = os.path.join(dir_name, "lc_tmp.txt")
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d.tolist(), [0.1, 0.2, 0.6])

            # Check the lc__index_ file.
            fname = os.path.join(dir_name, "lc_index_tmp.txt")
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=int)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0].tolist(), [0])
            self.assertEqual(data[1].tolist(), [0, 1])
            self.assertEqual(data[2].tolist(), [0, 1, 2])

            # Check the times_ file.
            fname = os.path.join(dir_name, "times_tmp.txt")
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0].tolist(), [0.0])
            self.assertEqual(data[1].tolist(), [0.0, 1.0])
            self.assertEqual(data[2].tolist(), [0.0, 1.0, 2.0])

            # Check that the other files exist.
            self.assertTrue(Path(os.path.join(dir_name, "filtered_likes_tmp.txt")).is_file())
            self.assertTrue(Path(os.path.join(dir_name, "ps_tmp.txt")).is_file())
            self.assertTrue(Path(os.path.join(dir_name, "all_ps_tmp.npy")).is_file())

    def test_save_and_load_results(self):
        times = [0.0, 10.0, 21.0, 30.5]
        num_times = len(times)

        # Fill the ResultList with 3 fake rows.
        rs = ResultList(times)
        for i in range(3):
            row = ResultRow(Trajectory(), num_times)
            row.set_psi_phi([0.1, 0.6, 0.2, float(i)], [2.0, 0.5, float(i), 1.0])
            row.filter_indices([t for t in range(num_times - i)])
            row.stamp = np.array([[float(i), float(i) / 3.0], [1.0, 0.5]])
            rs.append_result(row)

        # Try outputting the ResultList
        with tempfile.TemporaryDirectory() as dir_name:
            rs.save_to_files(dir_name, "tmp")

            # Load the results into a new data structure.
            rs2 = load_result_list_from_files(dir_name, "tmp")
            self.assertEqual(rs.num_results(), rs2.num_results())

            # Check the values match the original ResultSet.
            for i in range(rs.num_results()):
                row1 = rs.results[i]
                row2 = rs2.results[i]
                self.assertEqual(row1.num_times, row2.num_times)
                self.assertEqual(row1.valid_indices, row2.valid_indices)
                self.assertAlmostEqual(row1.final_likelihood, row2.final_likelihood)

                # Check psi, phi, and lc.
                row1_lc = row1.light_curve
                row2_lc = row2.light_curve
                for d in range(num_times):
                    self.assertAlmostEqual(row1.psi_curve[d], row2.psi_curve[d])
                    self.assertAlmostEqual(row1.phi_curve[d], row2.phi_curve[d])
                    self.assertAlmostEqual(row1_lc[d], row2_lc[d])

                # Check stamps.
                r1_stamp = row1.stamp.reshape(4)
                for d, v in enumerate(r1_stamp):
                    self.assertAlmostEqual(v, row2.stamp[d], delta=1e-3)
                self.assertIsNone(row2.all_stamps)

    def test_save_and_load_results_filtered(self):
        times = [0.0, 10.0, 21.0, 30.5]
        num_times = len(times)

        # Fill the ResultList with 5 fake rows.
        rs = ResultList(times, track_filtered=True)
        for i in range(5):
            trj = Trajectory()
            trj.x = 10 * i
            row = ResultRow(trj, num_times)
            row.set_psi_phi([0.1, 0.6, 0.2, float(i)], [2.0, 0.5, float(i), 1.0])
            row.filter_indices([t for t in range(num_times - i)])
            row.stamp = np.array([[float(i), float(i) / 3.0], [1.0, 0.5]])
            rs.append_result(row)

        # Filter out one result with label "test".
        rs.filter_results([0, 2, 3, 4], "test")

        # Filter out a second result with label "test2".
        rs.filter_results([0, 1], "test2")

        # Try outputting the ResultList
        with tempfile.TemporaryDirectory() as dir_name:
            rs.save_to_files(dir_name, "tmp")

            # Check that the filtered file for "test" exists.
            fname1 = os.path.join(dir_name, f"filtered_results_test_tmp.txt")
            self.assertTrue(Path(fname1).is_file())

            # Load and check the results.
            trjs = FileUtils.load_results_file_as_trajectories(fname1)
            self.assertEqual(len(trjs), 1)
            self.assertEqual(trjs[0].x, 10)

            # Check that the filtered file for "test2" exists.
            fname2 = os.path.join(dir_name, f"filtered_results_test2_tmp.txt")
            self.assertTrue(Path(fname2).is_file())

            # Load and check the results.
            trjs = FileUtils.load_results_file_as_trajectories(fname2)
            self.assertEqual(len(trjs), 2)
            self.assertEqual(trjs[0].x, 30)
            self.assertEqual(trjs[1].x, 40)


if __name__ == "__main__":
    unittest.main()
