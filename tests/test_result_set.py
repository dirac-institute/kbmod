import unittest

from kbmod.analysis_utils import *
from kbmod.result_set import *
from kbmod.search import *

class test_result_set(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]

    def test_append_single(self):
        rs = ResultSet()
        self.assertEqual(rs.num_results(), 0)

        for i in range(5):
            t = trajectory()
            t.lh = float(i)
            rs.append_result(ResultDataRow(t, self.times))
        self.assertEqual(rs.num_results(), 5)

        for i in range(5):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].final_lh, float(i))

    def test_append_list(self):
        rs = ResultSet()
        self.assertEqual(rs.num_results(), 0)

        # Create a list of rows to append.
        res_list = []
        for i in range(5):
            t = trajectory()
            t.lh = float(i)
            res_list.append(ResultDataRow(t, self.times))

        # Append the rows in one batch and check that it worked.
        rs.append_result_list(res_list)
        self.assertEqual(rs.num_results(), 5)
        for i in range(5):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].final_lh, float(i))

    def test_append_result_set(self):
        rs1 = ResultSet()
        rs2 = ResultSet()
        self.assertEqual(rs1.num_results(), 0)
        self.assertEqual(rs2.num_results(), 0)

        # Fill the first ResultSet with 5 rows.
        for i in range(5):
            t = trajectory()
            t.lh = float(i)
            rs1.append_result(ResultDataRow(t, self.times))

        # Fill a second Result set with 5 different rows.
        for i in range(5):
            t = trajectory()
            t.lh = float(i) + 5.0
            rs2.append_result(ResultDataRow(t, self.times))

        # Check that each result set has 5 results.
        self.assertEqual(rs1.num_results(), 5)
        self.assertEqual(rs2.num_results(), 5)

        # Append the two results set and check the 10 combined results.
        rs1.append_result_set(rs2)
        self.assertEqual(rs1.num_results(), 10)
        self.assertEqual(rs2.num_results(), 5)
        for i in range(10):
            self.assertIsNotNone(rs1.results[i].trajectory)
            self.assertEqual(rs1.results[i].final_lh, float(i))

    def test_clear(self):
        rs = ResultSet()
        for i in range(3):
            t = trajectory()
            rs.append_result(ResultDataRow(t, self.times))
        self.assertEqual(rs.num_results(), 3)

        rs.clear()
        self.assertEqual(rs.num_results(), 0)

    def test_fill_from_dictionary(self):
        # Generate a fake dictionary of results.
        st = SharedTools()
        keep = st.gen_results_dict()
        for i in range(5):
            keep["results"].append(trajectory())
            keep["stamps"].append([1])
            keep["new_lh"].append(float(i))
            keep["times"].append([10.0, 11.0, 12.0])
            keep["lc"].append([float(i) + 1.0])
            keep["lc_index"].append([0, 1, 2])
            keep["psi_curves"].append([1.0, 1.1, 1.2])

        # Append the dictionary's results to the ResultSet.
        rs = ResultSet()
        rs.append_result_dict(keep)
        self.assertEqual(rs.num_results(), 5)

        # Check that the correct results are stored.
        for i in range(5):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].lc, [float(i) + 1.0])
            self.assertEqual(rs.results[i].stamp, [1])
            self.assertEqual(rs.results[i].final_lh, float(i))
            self.assertEqual(rs.results[i].valid_times, [10.0, 11.0, 12.0])
            self.assertEqual(rs.results[i].valid_indices, [0, 1, 2])
            self.assertEqual(rs.results[i].all_stamps, None)
            self.assertEqual(rs.results[i].psi_curve, [1.0, 1.1, 1.2])
            self.assertEqual(rs.results[i].phi_curve, None)

    def test_fill_some_from_dictionary(self):
        # Generate a fake dictionary of results.
        st = SharedTools()
        keep = st.gen_results_dict()
        for i in range(5):
            keep["results"].append(trajectory())
            keep["stamps"].append([1])
            keep["new_lh"].append(float(i))
            keep["times"].append([10.0, 11.0, 12.0])
            keep["lc"].append(float(i) + 1.0)
            keep["lc_index"].append([0, 1, 2])
            keep["psi_curves"].append([1.0, 1.1, 1.2])

        # Only keep 3 of the results
        keep["final_results"] = [0, 2, 4]

        # Because of how the result dictionary is structured, stamps and all_stamps
        # do not use the final_results and must be filtered manually.
        keep["stamps"] = [keep["stamps"][i] for i in [0, 2, 4]]

        # Append the dictionary's results to the ResultSet.
        rs = ResultSet()
        rs.append_result_dict(keep)
        self.assertEqual(rs.num_results(), 3)

        # Check that the correct results are stored.
        self.assertEqual(rs.results[0].lc, 1.0)
        self.assertEqual(rs.results[1].lc, 3.0)
        self.assertEqual(rs.results[2].lc, 5.0)

    def test_fill_dictionary(self):
        # Fill the ResultSet with 4 fake rows.
        rs = ResultSet()
        for i in range(4):
            t = trajectory()
            t.lh = float(i)

            row = ResultDataRow(t, [0.0, 1.0, 2.0])
            row.final_lh = float(i) + 2.0
            row.stamp = [[i] * 2] * 2
            row.valid_indices = [0, 1, 3]
            row.phi_curve = [0.1, 0.1, 0.1]
            rs.append_result(row)

        # Generate the dictionary
        keep = rs.to_result_dict()
        self.assertEqual(len(keep["results"]), 4)
        self.assertEqual(len(keep["final_results"]), 4)
        for i in range(4):
            self.assertIsNotNone(keep["results"][i])
            self.assertEqual(keep["stamps"][i], [[i] * 2] * 2)
            self.assertEqual(keep["new_lh"][i], float(i) + 2.0)
            self.assertEqual(keep["times"][i], [0.0, 1.0, 2.0])
            self.assertEqual(keep["lc_index"][i], [0, 1, 3])
            self.assertEqual(keep["phi_curves"][i], [0.1, 0.1, 0.1])
            self.assertEqual(keep["final_results"][i], i)
        self.assertEqual(len(keep["lc"]), 0)
        self.assertEqual(len(keep["psi_curves"]), 0)
        self.assertEqual(len(keep["all_stamps"]), 0)

    def test_filter(self):
        rs = ResultSet()
        for i in range(10):
            t = trajectory()
            t.lh = float(i)
            rs.append_result(ResultDataRow(t, self.times))
        self.assertEqual(rs.num_results(), 10)

        # Do the filtering and check we have the correct ones.
        inds = [0, 2, 6, 7]
        rs.filter_results(inds)
        self.assertEqual(rs.num_results(), len(inds))
        for i in range(len(inds)):
            self.assertIsNotNone(rs.results[i].trajectory)
            self.assertEqual(rs.results[i].final_lh, float(inds[i]))

if __name__ == "__main__":
    unittest.main()
