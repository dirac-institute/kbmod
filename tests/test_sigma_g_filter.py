import numpy as np
import unittest

from kbmod.filters.sigma_g_filter import SigmaGClipping, apply_clipped_sigma_g
from kbmod.result_list import ResultRow, ResultList
from kbmod.results import Results
from kbmod.search import Trajectory


class test_sigma_g_math(unittest.TestCase):
    def test_create(self):
        params = SigmaGClipping()
        self.assertEqual(params.low_bnd, 25.0)
        self.assertEqual(params.high_bnd, 75.0)
        self.assertEqual(params.n_sigma, 2.0)
        self.assertFalse(params.clip_negative)
        self.assertAlmostEqual(params.coeff, 0.7413, places=4)

        self.assertRaises(ValueError, SigmaGClipping, n_sigma=-1.0)
        self.assertRaises(ValueError, SigmaGClipping, low_bnd=90.0, high_bnd=10.0)
        self.assertRaises(ValueError, SigmaGClipping, high_bnd=101.0)
        self.assertRaises(ValueError, SigmaGClipping, low_bnd=-1.0)

    def test_sigma_g_clipping(self):
        num_points = 20
        params = SigmaGClipping()

        # Everything is good.
        lh = np.array([(10.0 + i * 0.05) for i in range(num_points)])
        result = params.compute_clipped_sigma_g(lh)
        for i in range(num_points):
            self.assertTrue(i in result)

        # Two outliers
        lh[2] = 100.0
        lh[14] = -100.0
        result = params.compute_clipped_sigma_g(lh)
        for i in range(num_points):
            self.assertEqual(i in result, i != 2 and i != 14)

        # Add a third outlier
        lh[0] = 50.0
        result = params.compute_clipped_sigma_g(lh)
        for i in range(num_points):
            self.assertEqual(i in result, i != 0 and i != 2 and i != 14)

    def test_sigma_g_clipping_matrix(self):
        # Create a matrix with 5 lightcurves and 20 time steps.
        lh = np.array([[(10.0 + i * 0.05) for i in range(20)] for _ in range(5)])

        # Mark some points as obvious outliers.
        lh[1, 2] = 100.0
        lh[1, 14] = -100.0
        lh[2, 0] = 50.0
        lh[3, 2] = 100.0
        lh[3, 14] = -100.0
        lh[3, 0] = 50.0

        # "Mask" a few points in the final light curve with NANs
        lh[4, 7] = np.NAN
        lh[4, 8] = np.NAN
        lh[4, 11] = np.NAN

        expected = np.isfinite(lh) & (lh < 20.0) & (lh > 0.0)
        sigma_g = SigmaGClipping()
        index_valid = sigma_g.compute_clipped_sigma_g_matrix(lh)
        self.assertTrue(np.array_equal(index_valid, expected))

    def test_sigma_g_clipping_matrix_same(self):
        # Create a matrix with all the same values within a likelihood curve.
        lh = np.array([[5 for i in range(10)], [5.1 for i in range(10)]])
        expected = np.full(lh.shape, True)

        sigma_g = SigmaGClipping()
        index_valid = sigma_g.compute_clipped_sigma_g_matrix(lh)
        self.assertTrue(np.array_equal(index_valid, expected))

    def test_sigma_g_negative_clipping(self):
        num_points = 20
        lh = np.array([(-1.0 + i * 0.2) for i in range(num_points)])
        lh[2] = 20.0
        lh[14] = -20.0

        params = SigmaGClipping(clip_negative=True)
        result = params.compute_clipped_sigma_g(lh)
        for i in range(num_points):
            self.assertEqual(i in result, i > 2 and i != 5 and i != 14)

    def test_sigma_g_all_negative_clipping(self):
        num_points = 10
        lh = np.array([(-100.0 + i * 0.2) for i in range(num_points)])
        params = SigmaGClipping(clip_negative=True)
        result = params.compute_clipped_sigma_g(lh)
        self.assertEqual(len(result), 0)

    def test_sigma_g_clipping_matrix_negative_clipping(self):
        # Create a matrix with all the same values within a likelihood curve.
        num_points = 20
        lh = np.array(
            [
                [5 for i in range(num_points)],
                [(-1.0 + i * 0.2) for i in range(num_points)],  # Half negative
                [(-100.0 + i * 0.2) for i in range(num_points)],  # all negative
            ]
        )
        expected = np.array(
            [
                [True] * num_points,
                [False, False, False, True, True, False] + [True] * (num_points - 6),
                [False] * num_points,
            ]
        )

        sigma_g = SigmaGClipping(clip_negative=True)
        index_valid = sigma_g.compute_clipped_sigma_g_matrix(lh)
        self.assertTrue(np.array_equal(index_valid, expected))

    def test_apply_clipped_sigma_g(self):
        """Confirm the clipped sigmaG filter works when used in the bulk filter mode."""
        num_times = 20
        times = [(10.0 + 0.1 * float(i)) for i in range(num_times)]
        r_set = ResultList(times, track_filtered=True)
        phi_all = np.array([0.1] * num_times)

        for i in range(5):
            row = ResultRow(Trajectory(), num_times)
            psi_i = np.array([1.0] * num_times)
            for j in range(i):
                psi_i[j] = 100.0
            row.set_psi_phi(psi_i, phi_all)
            r_set.append_result(row)

        clipper = SigmaGClipping(10, 90)
        apply_clipped_sigma_g(clipper, r_set, num_threads=2)
        self.assertEqual(r_set.num_results(), 5)

        # Confirm that the ResultRows were modified in place.
        for i in range(5):
            self.assertEqual(len(r_set.results[i].valid_indices), num_times - i)

    def test_apply_clipped_sigma_g_results(self):
        """Confirm the clipped sigmaG filter works when used with a Results object."""
        num_times = 20
        num_results = 5
        trj_all = [Trajectory() for _ in range(num_results)]
        table = Results.from_trajectories(trj_all)

        phi_all = np.full((num_results, num_times), 0.1)
        psi_all = np.full((num_results, num_times), 1.0)
        for i in range(5):
            for j in range(i):
                psi_all[i, j] = 100.0
        table.add_psi_phi_data(psi_all, phi_all)

        clipper = SigmaGClipping(10, 90)
        apply_clipped_sigma_g(clipper, table)
        self.assertEqual(len(table), 5)

        # Confirm that the ResultRows were modified in place.
        for i in range(num_results):
            valid = table["obs_valid"][i]
            for j in range(i):
                self.assertFalse(valid[j])
            for j in range(i, num_times):
                self.assertTrue(valid[j])

    def test_sigmag_computation(self):
        self.assertAlmostEqual(SigmaGClipping.find_sigma_g_coeff(25.0, 75.0), 0.7413, delta=0.001)
        self.assertRaises(ValueError, SigmaGClipping.find_sigma_g_coeff, -1.0, 75.0)
        self.assertRaises(ValueError, SigmaGClipping.find_sigma_g_coeff, 25.0, 110.0)
        self.assertRaises(ValueError, SigmaGClipping.find_sigma_g_coeff, 75.0, 25.0)


if __name__ == "__main__":
    unittest.main()
