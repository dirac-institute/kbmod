import unittest

import numpy as np

from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.search import HAS_GPU, Trajectory
from kbmod.trajectory_explorer import TrajectoryExplorer


class test_trajectory_explorer(unittest.TestCase):
    def setUp(self):
        # image properties
        self.img_count = 20
        self.dim_x = 120
        self.dim_y = 115

        # create a Trajectory for the object
        self.x0 = 27
        self.y0 = 50
        self.vx = 21.0
        self.vy = -5.0
        self.trj = Trajectory(self.x0, self.y0, self.vx, self.vy, flux=500.0)

        # create image set with single moving object
        fake_times = [i / self.img_count for i in range(self.img_count)]
        fake_ds = FakeDataSet(
            self.dim_x,
            self.dim_y,
            fake_times,
            noise_level=2.0,
            psf_val=1.0,
            use_seed=True,
        )
        fake_ds.insert_object(self.trj)

        # Remove at least observation from the trajectory.
        pred_x = self.trj.get_x_index(fake_times[10])
        pred_y = self.trj.get_y_index(fake_times[10])
        sci_t10 = fake_ds.stack.get_single_image(10).get_science()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                sci_t10.set_pixel(pred_y + dy, pred_x + dx, 0.0001)

        self.explorer = TrajectoryExplorer(fake_ds.stack)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_evaluate_trajectory(self):
        result = self.explorer.evaluate_linear_trajectory(self.x0, self.y0, self.vx, self.vy)

        # We found the trajectory we were loooking for.
        self.assertEqual(len(result), 1)
        self.assertEqual(result["x"][0], self.x0)
        self.assertEqual(result["y"][0], self.y0)
        self.assertEqual(result["vx"][0], self.vx)
        self.assertEqual(result["vy"][0], self.vy)

        # The statistics seem reasonable.
        self.assertGreater(result["likelihood"][0], 50.0)
        self.assertGreater(result["flux"][0], 50.0)
        self.assertGreater(result["obs_count"][0], 10)

        # We compute the rest of the data we need.
        self.assertEqual(len(result["obs_valid"][0]), 20)
        self.assertEqual(len(result["psi_curve"][0]), 20)
        self.assertEqual(len(result["phi_curve"][0]), 20)
        self.assertEqual(len(result["all_stamps"][0]), 20)

        # Check that we have the correct stamp data.
        width = 2 * self.explorer.config["stamp_radius"] + 1
        self.assertTrue("coadd_sum" in result.colnames)
        self.assertEqual(result["coadd_sum"][0].shape, (width, width))
        self.assertTrue("coadd_mean" in result.colnames)
        self.assertEqual(result["coadd_mean"][0].shape, (width, width))
        self.assertTrue("coadd_median" in result.colnames)
        self.assertEqual(result["coadd_median"][0].shape, (width, width))
        self.assertTrue("all_stamps" in result.colnames)
        self.assertEqual(result["all_stamps"][0].shape, (self.img_count, width, width))

        # At least one index 10 should be filtered by sigma G filtering.
        self.explorer.apply_sigma_g(result)
        self.assertFalse(result["obs_valid"][0][10])


if __name__ == "__main__":
    unittest.main()
