import numpy as np
import unittest

from kbmod.fake_data.fake_result_creator import (
    add_fake_psi_phi_to_results,
    add_fake_coadds_to_results,
    make_fake_in_image_trajectory_info,
    make_fake_results,
    make_fake_trajectories,
)


class test_fake_result_creator(unittest.TestCase):
    def test_create_make_fake_in_image_trajectory_info(self):
        height = 200
        width = 250
        dt = 10.0
        x0, vx, y0, vy = make_fake_in_image_trajectory_info(100, height, width, dt=dt)

        # We generate the correct number of samples.
        self.assertEqual(len(x0), 100)
        self.assertEqual(len(vx), 100)
        self.assertEqual(len(y0), 100)
        self.assertEqual(len(vy), 100)

        # The samples are mostly different.
        self.assertGreater(len(np.unique(x0)), 50)
        self.assertGreater(len(np.unique(y0)), 50)
        self.assertGreater(len(np.unique(vx)), 50)
        self.assertGreater(len(np.unique(vy)), 50)

        # The starting positions are within the imae bounds.
        self.assertTrue(np.all(x0 >= 0))
        self.assertTrue(np.all(x0 < width))
        self.assertTrue(np.all(y0 >= 0))
        self.assertTrue(np.all(y0 < height))

        # The end positions are within the image bounds.
        xe = x0 + vx * dt
        ye = y0 + vy * dt
        self.assertTrue(np.all(xe >= 0))
        self.assertTrue(np.all(xe < width))
        self.assertTrue(np.all(ye >= 0))
        self.assertTrue(np.all(ye < height))

    def test_make_fake_trajectories(self):
        height = 200
        width = 250
        dt = 10.0
        trjs = make_fake_trajectories(
            100,
            height,
            width,
            dt=dt,
            min_lh=0.0,
            max_lh=20.0,
            min_flux=0.0,
            max_flux=250.0,
            num_obs=5,
        )
        self.assertEqual(len(trjs), 100)

        for trj in trjs:
            self.assertGreaterEqual(trj.x, 0)
            self.assertLess(trj.x, width)
            self.assertGreaterEqual(trj.y, 0)
            self.assertLess(trj.y, height)
            self.assertGreaterEqual(trj.lh, 0.0)
            self.assertLessEqual(trj.lh, 20.0)
            self.assertGreaterEqual(trj.flux, 0.0)
            self.assertLessEqual(trj.flux, 250.0)
            self.assertEqual(trj.obs_count, 5)

    def test_make_fake_results(self):
        num_results = 100
        num_times = 10

        results = make_fake_results(num_times, 200, 250, num_results)
        self.assertEqual(len(results), 100)
        self.assertTrue("x" in results.colnames)
        self.assertTrue("y" in results.colnames)
        self.assertTrue("vx" in results.colnames)
        self.assertTrue("vy" in results.colnames)
        self.assertTrue("likelihood" in results.colnames)
        self.assertTrue("flux" in results.colnames)
        self.assertTrue("obs_count" in results.colnames)
        self.assertFalse("psi_curve" in results.colnames)
        self.assertFalse("phi_curve" in results.colnames)
        self.assertFalse("coadd_mean" in results.colnames)
        self.assertFalse("coadd_median" in results.colnames)
        self.assertFalse("coadd_sum" in results.colnames)
        self.assertEqual(len(results.mjd_utc_mid), num_times)

        # Test that we can add fake psi and phi curves to the results.
        results = add_fake_psi_phi_to_results(results)
        self.assertTrue("psi_curve" in results.colnames)
        self.assertTrue("phi_curve" in results.colnames)
        for i in range(num_results):
            self.assertEqual(len(results["psi_curve"][i]), num_times)
            self.assertEqual(len(results["phi_curve"][i]), num_times)
            self.assertTrue(np.all(results["psi_curve"][i] >= 0))
            self.assertTrue(np.all(results["phi_curve"][i] > 0))

        # Test that we can add fake stamps to the results
        results = add_fake_coadds_to_results(results, "mean", 3)
        results = add_fake_coadds_to_results(results, "median", 10)
        self.assertTrue("coadd_mean" in results.colnames)
        self.assertTrue("coadd_median" in results.colnames)
        for i in range(num_results):
            self.assertEqual(results["coadd_mean"][i].shape, (7, 7))
            self.assertEqual(results["coadd_median"][i].shape, (21, 21))

    def test_add_fake_psi_phi_to_results(self):
        num_results = 100
        num_times = 10
        num_pts = num_results * num_times

        results = make_fake_results(num_times, 200, 250, num_results)

        # Add default fake psi and phi curves.
        results = add_fake_psi_phi_to_results(results, signal_mean=10.0, data_var=0.5)
        self.assertTrue(np.all(np.abs(results["psi_curve"] - 20.0) < 2.0))
        self.assertTrue(np.all(np.abs(results["phi_curve"] - 2.0) < 0.5))
        self.assertTrue(np.all(results["obs_valid"]))

        # Add fake psi and phi curves with masking.
        results = make_fake_results(num_times, 200, 250, num_results)
        results = add_fake_psi_phi_to_results(results, masked_fraction=0.2)
        valid = results["obs_valid"]

        self.assertFalse(np.any(np.isnan(results["psi_curve"][valid])))
        self.assertFalse(np.any(np.isnan(results["phi_curve"][valid])))
        self.assertTrue(np.all(np.isnan(results["psi_curve"][~valid])))
        self.assertTrue(np.all(np.isnan(results["psi_curve"][~valid])))

        self.assertAlmostEqual(np.sum(valid) / num_pts, 0.8, delta=0.1)
        self.assertAlmostEqual(np.mean(results["psi_curve"][valid]), 20.0, delta=4.0)
        self.assertAlmostEqual(np.mean(results["phi_curve"][valid]), 2.0, delta=0.5)

        # Add fake psi and phi curves with outliers.
        results = make_fake_results(num_times, 200, 250, num_results)
        results = add_fake_psi_phi_to_results(
            results,
            signal_mean=10.0,
            data_var=0.5,
            outlier_fraction=0.3,
            outlier_mean=100.0,
            masked_fraction=0.0,
        )
        not_outlier = results["psi_curve"] < 50.0
        self.assertTrue(np.all(not_outlier == results["obs_valid"]))
        self.assertAlmostEqual(np.mean(results["psi_curve"][not_outlier]), 20.0, delta=4.0)
        self.assertAlmostEqual(np.mean(results["psi_curve"][~not_outlier]), 100.0, delta=10.0)
        self.assertAlmostEqual(np.mean(results["phi_curve"]), 2.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
