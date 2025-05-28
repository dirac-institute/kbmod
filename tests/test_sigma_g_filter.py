import numpy as np
import unittest
import warnings

from kbmod.filters.sigma_g_filter import SigmaGClipping, apply_clipped_sigma_g
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
        lh[4, 7] = np.nan
        lh[4, 8] = np.nan
        lh[4, 11] = np.nan

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
            self.assertEqual(i in result, i > 2 and i != 14)

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
                [False, False, False] + [True] * (num_points - 3),
                [False] * num_points,
            ]
        )

        sigma_g = SigmaGClipping(clip_negative=True)

        # Surpress the warning we get from encountering a row of all NaNs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index_valid = sigma_g.compute_clipped_sigma_g_matrix(lh)
        self.assertTrue(np.array_equal(index_valid, expected))

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

    def test_apply_clipped_sigma_g_empty(self):
        """Confirm the clipped sigmaG filter works when used with an empty Results object."""
        table = Results()
        self.assertEqual(len(table), 0)

        psi_all = np.array([]).reshape((0, 10))
        phi_all = np.array([]).reshape((0, 10))
        table.add_psi_phi_data(psi_all, phi_all)
        self.assertTrue("psi_curve" in table.colnames)
        self.assertTrue("phi_curve" in table.colnames)

        clipper = SigmaGClipping(10, 90)
        apply_clipped_sigma_g(clipper, table)
        self.assertEqual(len(table), 0)

    def test_sigmag_parity(self):
        """Test that we get the same results when using the batch and the non-batch methods."""
        num_tests = 20

        # Run the test with differing numbers of points and with/without clipping.
        for num_obs in [10, 20, 50]:
            for clipped in [True, False]:
                for num_extreme in [0, 1, 2, 3]:
                    with self.subTest(
                        num_obs_used=num_obs, use_clipped=clipped, num_extreme_used=num_extreme
                    ):
                        # Generate the data from a fixed random seed (same for every subtest).
                        rng = np.random.default_rng(100)
                        data = 10.0 * rng.random((num_tests, num_obs)) - 0.5

                        # Add extreme values for each row.
                        for row in range(num_tests):
                            for ext_num in range(num_extreme):
                                idx = int(num_obs * rng.random())
                                data[row, idx] = 100.0 * rng.random() - 50.0

                            clipper = SigmaGClipping(25, 75, clip_negative=clipped)

                        batch_res = clipper.compute_clipped_sigma_g_matrix(data)
                        for row in range(num_tests):
                            # Compute the individual results (as indices) and convert
                            # those into a vector of bools for comparison.
                            ind_res = clipper.compute_clipped_sigma_g(data[row])
                            ind_bools = [(idx in ind_res) for idx in range(num_obs)]
                            self.assertTrue(np.array_equal(batch_res[row], ind_bools))

    def test_sigmag_computation(self):
        self.assertAlmostEqual(SigmaGClipping.find_sigma_g_coeff(25.0, 75.0), 0.7413, delta=0.001)
        self.assertRaises(ValueError, SigmaGClipping.find_sigma_g_coeff, -1.0, 75.0)
        self.assertRaises(ValueError, SigmaGClipping.find_sigma_g_coeff, 25.0, 110.0)
        self.assertRaises(ValueError, SigmaGClipping.find_sigma_g_coeff, 75.0, 25.0)


if __name__ == "__main__":
    unittest.main()
