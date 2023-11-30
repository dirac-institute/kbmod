import os
import numpy as np
import tempfile
import unittest
from pathlib import Path

from kbmod.analysis_utils import *
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

        example_stample = np.ones((5, 5))
        self.rdr.all_stamps = np.array([np.copy(example_stample) for _ in range(4)])

    def test_get_boolean_valid_indices(self):
        self.assertEqual(self.rdr.valid_indices_as_booleans(), [True, True, True, True])

        self.rdr.filter_indices([1, 2])
        self.assertEqual(self.rdr.valid_indices_as_booleans(), [False, True, True, False])

    def test_filter(self):
        self.assertEqual(self.rdr.valid_indices, [0, 1, 2, 3])
        self.assertEqual(self.rdr.valid_times(self.times), [1.0, 2.0, 3.0, 4.0])
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
        self.assertEqual(self.rdr.psi_curve, [1.0, 1.1, 1.2, 1.3])
        self.assertEqual(self.rdr.phi_curve, [1.0, 1.0, 0.0, 2.0])
        self.assertEqual(self.rdr.all_stamps.shape, (4, 5, 5))

    def test_set_psi_phi(self):
        self.rdr.set_psi_phi([1.5, 1.1, 1.2, 1.0], [1.0, 0.0, 0.0, 0.5])
        self.assertEqual(self.rdr.psi_curve, [1.5, 1.1, 1.2, 1.0])
        self.assertEqual(self.rdr.phi_curve, [1.0, 0.0, 0.0, 0.5])
        self.assertEqual(self.rdr.light_curve, [1.5, 0.0, 0.0, 2.0])

    def test_compute_likelihood_curve(self):
        self.rdr.set_psi_phi([1.5, 1.1, 1.2, 1.1], [1.0, 0.0, 4.0, 0.25])
        lh = self.rdr.likelihood_curve
        self.assertEqual(lh, [1.5, 0.0, 0.6, 2.2])

    def test_to_from_yaml(self):
        yaml_str = self.rdr.to_yaml()
        self.assertGreater(len(yaml_str), 0)

        row2 = ResultRow.from_yaml(yaml_str)
        self.assertAlmostEqual(row2.final_likelihood, 2.3)
        self.assertEqual(row2.valid_indices, [0, 1, 2, 3])
        self.assertEqual(row2.valid_times(self.times), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(row2.trajectory.obs_count, 4)
        self.assertIsNone(row2.stamp)
        self.assertIsNotNone(row2.all_stamps)
        self.assertEqual(row2.all_stamps.shape[0], 4)
        self.assertEqual(row2.all_stamps.shape[1], 5)
        self.assertEqual(row2.all_stamps.shape[2], 5)

        self.assertIsNotNone(row2.trajectory)
        self.assertAlmostEqual(row2.trajectory.flux, 1.15)
        self.assertAlmostEqual(row2.trajectory.lh, 2.3)


class test_result_list(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

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
            t = Trajectory()
            t.x = i
            rs.append_result(ResultRow(t, self.num_times))
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

    def test_to_from_yaml(self):
        rs = ResultList(self.times, track_filtered=True)
        for i in range(10):
            row = ResultRow(Trajectory(), self.num_times)
            row.set_psi_phi(np.array([i] * self.num_times), np.array([0.01 * i] * self.num_times))
            rs.append_result(row)

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

    def test_save_results(self):
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
            fname = f"{dir_name}/results_tmp.txt"
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_results_file_as_trajectories(fname)
            self.assertEqual(len(data), 3)

            # Check the psi_ file.
            fname = f"{dir_name}/psi_tmp.txt"
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d.tolist(), [0.1, 0.2, 0.3])

            # Check the phi_ file.
            fname = f"{dir_name}/phi_tmp.txt"
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d.tolist(), [1.0, 1.0, 0.5])

            # Check the lc_ file.
            fname = f"{dir_name}/lc_tmp.txt"
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            for d in data:
                self.assertEqual(d.tolist(), [0.1, 0.2, 0.6])

            # Check the lc__index_ file.
            fname = f"{dir_name}/lc_index_tmp.txt"
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=int)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0].tolist(), [0])
            self.assertEqual(data[1].tolist(), [0, 1])
            self.assertEqual(data[2].tolist(), [0, 1, 2])

            # Check the times_ file.
            fname = f"{dir_name}/times_tmp.txt"
            self.assertTrue(Path(fname).is_file())
            data = FileUtils.load_csv_to_list(fname, use_dtype=float)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0].tolist(), [0.0])
            self.assertEqual(data[1].tolist(), [0.0, 1.0])
            self.assertEqual(data[2].tolist(), [0.0, 1.0, 2.0])

            # Check that the other files exist.
            self.assertTrue(Path(f"{dir_name}/filtered_likes_tmp.txt").is_file())
            self.assertTrue(Path(f"{dir_name}/ps_tmp.txt").is_file())
            self.assertTrue(Path(f"{dir_name}/all_ps_tmp.npy").is_file())

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
