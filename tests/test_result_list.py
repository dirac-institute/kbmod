import tempfile
import unittest

from pathlib import Path

from kbmod.analysis_utils import *
from kbmod.file_utils import *
from kbmod.result_list import *
from kbmod.search import *


class test_result_data_row(unittest.TestCase):
    def setUp(self):
        self.trj = trajectory()

        self.times = [1.0, 2.0, 3.0, 4.0]
        self.num_times = len(self.times)
        self.rdr = ResultRow(self.trj, self.num_times)
        self.rdr.set_psi_phi([1.0, 1.1, 1.2, 1.3], [1.0, 1.0, 0.0, 2.0])
        self.rdr.all_stamps = [1.0, 1.0, 1.0, 1.0]

    def test_get_trj_result(self):
        res = self.rdr.trj_result
        self.assertEqual(res.get_valid_indices_list(), [0, 1, 2, 3])

        self.rdr.filter_indices([1, 2])
        res2 = self.rdr.trj_result
        self.assertEqual(res2.get_valid_indices_list(), [1, 2])

    def test_filter(self):
        self.rdr.filter_indices([0, 2, 3])
        self.assertEqual(self.rdr.valid_indices, [0, 2, 3])
        self.assertEqual(self.rdr.valid_times(self.times), [1.0, 3.0, 4.0])

        # The curves and stamps should not change.
        self.assertEqual(self.rdr.psi_curve, [1.0, 1.1, 1.2, 1.3])
        self.assertEqual(self.rdr.phi_curve, [1.0, 1.0, 0.0, 2.0])
        self.assertEqual(self.rdr.all_stamps, [1.0, 1.0, 1.0, 1.0])

    def test_set_psi_phi(self):
        self.rdr.set_psi_phi([1.5, 1.1, 1.2, 1.0], [1.0, 0.0, 0.0, 0.5])
        self.assertEqual(self.rdr.psi_curve, [1.5, 1.1, 1.2, 1.0])
        self.assertEqual(self.rdr.phi_curve, [1.0, 0.0, 0.0, 0.5])
        self.assertEqual(self.rdr.light_curve, [1.5, 0.0, 0.0, 2.0])

    def test_compute_likelihood_curve(self):
        self.rdr.set_psi_phi([1.5, 1.1, 1.2, 1.1], [1.0, 0.0, 4.0, 0.25])
        lh = self.rdr.likelihood_curve
        self.assertEqual(lh, [1.5, 0.0, 0.6, 2.2])


class test_result_list(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

    def test_append_single(self):
        rs = ResultList(self.times)
        self.assertEqual(rs.num_results(), 0)

        for i in range(5):
            t = trajectory()
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
            t = trajectory()
            t.lh = float(i)
            rs1.append_result(ResultRow(t, self.num_times))

        # Fill a second Result set with 5 different rows.
        for i in range(5):
            t = trajectory()
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
            t = trajectory()
            rs.append_result(ResultRow(t, self.num_times))
        self.assertEqual(rs.num_results(), 3)

        rs.clear()
        self.assertEqual(rs.num_results(), 0)

    def test_fill_from_dictionary(self):
        times = [10.0, 11.0, 12.0]

        # Generate a fake dictionary of results.
        st = SharedTools()
        keep = st.gen_results_dict()
        for i in range(5):
            keep["results"].append(trajectory())
            keep["stamps"].append([1])
            keep["new_lh"].append(float(i))
            keep["times"].append(times)
            keep["lc"].append([float(i) + 1.0])
            keep["lc_index"].append([0, 1, 2])
            keep["psi_curves"].append([1.0, 1.1, 1.2])

        # Append the dictionary's results to the ResultList.
        rs = ResultList(times)
        rs.append_result_dict(keep)
        self.assertEqual(rs.num_results(), 5)

        # Check that the correct results are stored.
        for i in range(5):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].stamp, [1])
            self.assertEqual(rs.results[i].final_likelihood, float(i))
            self.assertEqual(rs.results[i].valid_times(times), times)
            self.assertEqual(rs.results[i].valid_indices, [0, 1, 2])
            self.assertEqual(rs.results[i].all_stamps, None)
            self.assertEqual(rs.results[i].psi_curve, [1.0, 1.1, 1.2])
            self.assertEqual(rs.results[i].phi_curve, None)

    def test_fill_some_from_dictionary(self):
        times = [10.0, 11.0, 12.0]

        # Generate a fake dictionary of results.
        st = SharedTools()
        keep = st.gen_results_dict()
        for i in range(5):
            keep["results"].append(trajectory())
            keep["stamps"].append([1])
            keep["new_lh"].append(float(i))
            keep["times"].append(times)
            keep["lc_index"].append([0, 1, 2])
            keep["psi_curves"].append([1.0, 1.1, 1.2])

        # Only keep 3 of the results
        keep["final_results"] = [0, 2, 4]

        # Because of how the result dictionary is structured, stamps and all_stamps
        # do not use the final_results and must be filtered manually.
        keep["stamps"] = [keep["stamps"][i] for i in [0, 2, 4]]

        # Append the dictionary's results to the ResultList.
        rs = ResultList(times)
        rs.append_result_dict(keep)
        self.assertEqual(rs.num_results(), 3)

        # Check that the correct results are stored.
        self.assertEqual(rs.results[0].final_likelihood, 0.0)
        self.assertEqual(rs.results[1].final_likelihood, 2.0)
        self.assertEqual(rs.results[2].final_likelihood, 4.0)

    def test_fill_dictionary(self):
        times = [0.0, 1.0, 2.0]

        # Fill the ResultList with 4 fake rows.
        rs = ResultList(times)
        for i in range(4):
            t = trajectory()
            row = ResultRow(t, 3)
            row.stamp = [[i] * 2] * 2
            row.set_psi_phi([0.1, 0.2, 0.3], [1.0, 1.0, 1.0])
            row.filter_indices([0, 1, 2])
            rs.append_result(row)

        # Generate the dictionary
        keep = rs.to_result_dict()
        self.assertEqual(len(keep["results"]), 4)
        self.assertEqual(len(keep["final_results"]), 4)
        for i in range(4):
            self.assertIsNotNone(keep["results"][i])
            self.assertEqual(keep["stamps"][i], [[i] * 2] * 2)
            self.assertAlmostEqual(keep["new_lh"][i], 0.6 / math.sqrt(3.0))
            self.assertEqual(keep["times"][i], [0.0, 1.0, 2.0])
            self.assertEqual(keep["lc_index"][i], [0, 1, 2])
            self.assertEqual(keep["psi_curves"][i], [0.1, 0.2, 0.3])
            self.assertEqual(keep["phi_curves"][i], [1.0, 1.0, 1.0])
            self.assertEqual(keep["final_results"][i], i)
            self.assertEqual(keep["lc"][i], [0.1, 0.2, 0.3])
        self.assertEqual(len(keep["all_stamps"]), 0)

    def test_filter(self):
        rs = ResultList(self.times)
        for i in range(10):
            t = trajectory()
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

    def test_filter_likelihood(self):
        rs = ResultList(self.times)
        for i in range(10):
            t = trajectory()
            t.lh = float(i)
            rs.append_result(ResultRow(t, self.num_times))
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering and check we have the correct ones.
        rs.filter_on_stats(4.5, -1)
        self.assertEqual(rs.num_results(), 5)
        for i in range(rs.num_results()):
            self.assertGreater(rs.results[i].final_likelihood, 4.5)

    def test_filter_valid_indices(self):
        rs = ResultList(self.times)
        for i in range(10):
            t = trajectory()
            row = ResultRow(t, self.num_times)
            row.filter_indices([k for k in range(i)])
            rs.append_result(row)
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering and check we have the correct ones.
        rs.filter_on_stats(-1.0, 4)
        self.assertEqual(rs.num_results(), 6)
        for i in range(rs.num_results()):
            self.assertGreaterEqual(len(rs.results[i].valid_indices), 4)

    def test_save_results(self):
        times = [0.0, 1.0, 2.0]

        # Fill the ResultList with 3 fake rows.
        rs = ResultList(times)
        for i in range(3):
            row = ResultRow(trajectory(), 3)
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
            row = ResultRow(trajectory(), num_times)
            row.set_psi_phi([0.1, 0.6, 0.2, float(i)], [2.0, 0.5, float(i), 1.0])
            row.filter_indices([t for t in range(num_times - i)])
            row.stamp = np.array([[float(i), float(i)/3.0], [1.0, 0.5]])
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
                self.assertEqual(row1.stamp is None, row2.stamp is None)
                if row1.stamp is not None:
                    r1_stamp = row1.stamp.reshape(4)
                    for d, v in enumerate(r1_stamp):
                        self.assertAlmostEqual(v, row2.stamp[d], delta=1e-3)
                
                self.assertEqual(row1.all_stamps is None, row2.all_stamps is None)
                if row1.all_stamps is not None:
                    self.assertEqual(len(row1.all_stamps), len(row2.all_stamps))


if __name__ == "__main__":
    unittest.main()
