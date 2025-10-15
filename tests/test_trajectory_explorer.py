import unittest

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.results import Results
from kbmod.search import kb_has_gpu, Trajectory
from kbmod.trajectory_explorer import refine_all_results, TrajectoryExplorer


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
        fake_times = np.array([59000.0 + i / self.img_count for i in range(self.img_count)])
        self.fake_ds = FakeDataSet(
            self.dim_x,
            self.dim_y,
            fake_times,
            noise_level=2.0,
            psf_val=1.0,
            use_seed=101,
        )
        self.fake_ds.insert_object(self.trj)

        # Remove at least one observation from the trajectory.
        zeroed_times = fake_times - fake_times[0]
        pred_x = self.trj.get_x_index(zeroed_times[10])
        pred_y = self.trj.get_y_index(zeroed_times[10])
        sci_t10 = self.fake_ds.stack_py.sci[10]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                sci_t10[pred_y + dy, pred_x + dx] = 0.0001

        self.explorer = TrajectoryExplorer(self.fake_ds.stack_py)

    def test_evaluate_trajectory(self):
        result = self.explorer.evaluate_linear_trajectory(self.x0, self.y0, self.vx, self.vy, False)

        # We found the trajectory we were looking for.
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

    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_evaluate_trajectory_parity(self):
        """Test that we get the same results with GPU or CPU-only code."""
        config = SearchConfiguration()
        config.set("gpu_filter", False)
        explorer2 = TrajectoryExplorer(self.fake_ds.stack_py, config=config)
        result1 = explorer2.evaluate_linear_trajectory(self.x0, self.y0, self.vx, self.vy, True)
        result2 = explorer2.evaluate_linear_trajectory(self.x0, self.y0, self.vx, self.vy, False)

        # We found the trajectory we were loooking for.
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(result1["x"][0], result2["x"][0])
        self.assertEqual(result1["y"][0], result2["y"][0])
        self.assertAlmostEqual(result1["vx"][0], result2["vx"][0])
        self.assertAlmostEqual(result1["vy"][0], result2["vy"][0])
        self.assertAlmostEqual(result1["likelihood"][0], result2["likelihood"][0])
        self.assertAlmostEqual(result1["flux"][0], result2["flux"][0])
        self.assertAlmostEqual(result1["obs_count"][0], result2["obs_count"][0])

    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_evaluate_around_linear_trajectory(self):
        radius = 3
        edge_length = 2 * radius + 1
        num_pixels = edge_length * edge_length

        results = self.explorer.evaluate_around_linear_trajectory(
            self.x0,
            self.y0,
            self.vx,
            self.vy,
            pixel_radius=radius,
            max_ang_offset=0.2618,
            ang_step=0.035,
            max_vel_offset=10.0,
            vel_step=0.5,
        )

        # Using the above settings should provide 615 trajectories per starting pixel.
        self.assertEqual(len(results), num_pixels * 615)

        # Count the number of results we have per starting pixel.
        counts = np.zeros((edge_length, edge_length))
        for row in range(len(results)):
            self.assertGreaterEqual(results["x"][row], self.x0 - 3)
            self.assertLessEqual(results["x"][row], self.x0 + 3)
            self.assertGreaterEqual(results["y"][row], self.y0 - 3)
            self.assertLessEqual(results["y"][row], self.y0 + 3)

            x = results["x"][row] - self.x0 + 3
            y = results["y"][row] - self.y0 + 3
            counts[y, x] += 1
        self.assertTrue(np.all(counts == 615))

    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_refine_linear_trajectory(self):
        # Start with a trajectory that is offset from the true one.
        results = self.explorer.refine_linear_trajectory(
            self.x0 + 1,
            self.y0 - 1,
            self.vx + 2.5,
            self.vy - 2.5,
            pixel_radius=3,
            max_dv=10.0,
            dv_steps=21,
        )

        # We get one result (and it is the correct one).
        self.assertEqual(len(results), 1)
        self.assertEqual(results["x"][0], self.x0)
        self.assertEqual(results["y"][0], self.y0)
        self.assertLess(np.abs(results["vx"][0] - self.vx), 1.0)
        self.assertLess(np.abs(results["vy"][0] - self.vy), 1.0)


    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_refine_all_results(self):
        # Create a small fake data set.
        num_times = 5
        width = 500
        height = 500
        fake_times = [59000.0 + float(i) for i in range(num_times)]
        fake_ds = FakeDataSet(width, height, fake_times, psf_val=0.01)

        # Insert two fake objects.
        trj1 = Trajectory(x=17, y=12, vx=21.0, vy=16.0, flux=500.0)
        fake_ds.insert_object(trj1)

        trj2 = Trajectory(x=400, y=100, vx=-5.0, vy=10.0, flux=250.0)
        fake_ds.insert_object(trj2)

        # Create fake results around the true trajectories with noise and duplicates.
        results_list = [
            Trajectory(x=17, y=13, vx=21.0, vy=16.0, lh=10.0, obs_count=5),
            Trajectory(x=16, y=15, vx=20.0, vy=15.0, lh=10.0, obs_count=5),
            Trajectory(x=15, y=9, vx=22.0, vy=17.0, lh=10.0, obs_count=5),
            Trajectory(x=400, y=101, vx=-4.0, vy=11.0, lh=10.0, obs_count=5),
            Trajectory(x=401, y=99, vx=-6.0, vy=9.0, lh=10.0, obs_count=5),
            Trajectory(x=399, y=100, vx=-5.0, vy=10.0, lh=10.0, obs_count=5),
            Trajectory(x=400, y=100, vx=-15.0, vy=30.0, lh=10.0, obs_count=5),
        ]
        org_results = Results.from_trajectories(results_list)

        # Do the refining. We should get three results after deduplication.
        new_results = refine_all_results(org_results, fake_ds.stack_py)
        self.assertEqual(len(new_results), 3)

        # The first two refined results should be close to the true inserted
        # objects in order of their flux. The third result is junk, but is not
        # close to either of the first two (due to velocity difference).
        self.assertAlmostEqual(new_results["x"][0], trj1.x, delta=1.0)
        self.assertAlmostEqual(new_results["y"][0], trj1.y, delta=1.0)
        self.assertAlmostEqual(new_results["vx"][0], trj1.vx, delta=1.0)
        self.assertAlmostEqual(new_results["vy"][0], trj1.vy, delta=1.0)

        self.assertAlmostEqual(new_results["x"][1], trj2.x, delta=1.0)
        self.assertAlmostEqual(new_results["y"][1], trj2.y, delta=1.0)
        self.assertAlmostEqual(new_results["vx"][1], trj2.vx, delta=1.0)
        self.assertAlmostEqual(new_results["vy"][1], trj2.vy, delta=1.0)

        self.assertAlmostEqual(new_results["x"][2], 400, delta=1.0)
        self.assertAlmostEqual(new_results["y"][2], 100, delta=1.0)


if __name__ == "__main__":
    unittest.main()
