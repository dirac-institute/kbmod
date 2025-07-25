import numpy as np
import os
import tempfile
import unittest

from pathlib import Path

from kbmod.fake_data.fake_data_creator import *
from kbmod.trajectory_generator import VelocityGridSearch
from kbmod.wcs_utils import make_fake_wcs
from kbmod.work_unit import WorkUnit


class test_fake_image_creator(unittest.TestCase):
    def test_create_fake_times(self):
        times1 = create_fake_times(10, t0=0.0, obs_per_day=3, intra_night_gap=0.01, inter_night_gap=1)
        expected = [0.0, 0.01, 0.02, 1.0, 1.01, 1.02, 2.0, 2.01, 2.02, 3.0]
        self.assertEqual(len(times1), 10)
        for i in range(10):
            self.assertAlmostEqual(times1[i], expected[i])

        times2 = create_fake_times(7, t0=10.0, obs_per_day=1, intra_night_gap=0.5, inter_night_gap=2)
        expected = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
        self.assertEqual(len(times2), 7)
        for i in range(7):
            self.assertAlmostEqual(times2[i], expected[i])

    def test_make_fake_image_stack(self):
        """Test that we can create a fake ImageStackPy."""
        fake_times = np.arange(10)
        fake_stack = make_fake_image_stack(200, 300, fake_times)
        self.assertEqual(fake_stack.num_times, 10)
        self.assertEqual(fake_stack.width, 300)
        self.assertEqual(fake_stack.height, 200)
        self.assertEqual(fake_stack.npixels, 200 * 300)
        self.assertEqual(fake_stack.total_pixels, 10 * 200 * 300)
        self.assertTrue(np.all(fake_stack.num_masked_pixels() == 0))
        self.assertEqual(len(fake_stack.sci), 10)
        self.assertEqual(len(fake_stack.var), 10)
        for idx in range(10):
            self.assertEqual(fake_stack.sci[idx].shape, (200, 300))
            self.assertEqual(fake_stack.var[idx].shape, (200, 300))
            self.assertTrue(len(np.unique(fake_stack.sci[idx])) > 1)
            self.assertTrue(np.allclose(fake_stack.var[idx], 4.0))
        self.assertEqual(len(fake_stack.psfs), 10)

    def test_mask_fake_image_stack(self):
        """Test that we can add random masking to an ImageStackPy."""
        fake_times = np.arange(10)
        fake_stack = make_fake_image_stack(200, 300, fake_times)

        # Nothing is masked by default.
        self.assertTrue(np.all(fake_stack.num_masked_pixels() == 0))

        # Mask 10% of the pixels in each image.
        rng = np.random.default_rng(103)
        image_stack_add_random_masks(fake_stack, 0.1, rng=rng)
        self.assertTrue(np.all(fake_stack.num_masked_pixels() > 0))
        self.assertTrue(np.all(fake_stack.get_masked_fractions() > 0.05))
        self.assertTrue(np.all(fake_stack.get_masked_fractions() < 0.15))

    def test_image_stack_add_fake_object(self):
        """Test that we can insert a fake object into an ImageStackPy."""
        num_times = 5
        height = 200
        width = 300

        # Create a fake ImageStackPy with 5 time steps and a masked pixel at time 3.
        fake_times = np.arange(num_times)
        sci = np.full((num_times, height, width), 0.0)
        var = np.full((num_times, height, width), 1.0)
        sci[3][66, 53] = np.nan
        var[3][66, 53] = np.nan
        psfs = [PSF.from_gaussian(0.5) for i in range(num_times)]
        fake_stack = ImageStackPy(fake_times, sci, var, psfs=psfs)

        image_stack_add_fake_object(fake_stack, 50, 60, 1.0, 2.0, flux=100.0)
        for t_idx, t_val in enumerate(fake_times):
            # Check that we receive a signal at the correct location that
            # is non-zero but less than the flux (due to the PSF) at each time step.
            px = int(50 + t_val + 0.5)
            py = int(60 + 2.0 * t_val + 0.5)

            if t_idx == 3:
                # At time 3, the pixel is masked, so we should not have a signal.
                self.assertTrue(np.isnan(fake_stack.sci[t_idx][py, px]))
            else:
                self.assertGreater(fake_stack.sci[t_idx][py, px], 50.0)
                self.assertLess(fake_stack.sci[t_idx][py, px], 100.0)

            # Far away from the object, the signal should be zero.
            self.assertAlmostEqual(fake_stack.sci[t_idx][30, 40], 0.0)

    def test_image_stack_add_fake_object_quadratic(self):
        """Test that we can insert a fake object with a quadratic trajectory into an ImageStackPy."""
        num_times = 5
        height = 200
        width = 300

        fake_times = np.arange(num_times)
        sci = np.full((num_times, height, width), 0.0)
        var = np.full((num_times, height, width), 1.0)
        psfs = [PSF.from_gaussian(0.5) for i in range(num_times)]
        fake_stack = ImageStackPy(fake_times, sci, var, psfs=psfs)

        image_stack_add_fake_object(fake_stack, 50, 60, 1.0, 2.0, ax=1.1, ay=-0.5, flux=100.0)
        for t_idx, t_val in enumerate(fake_times):
            # Check that we receive a signal at the correct location that
            # is non-zero but less than the flux (due to the PSF) at each time step.
            px = int(50 + t_val + 0.5 * 1.1 * t_val * t_val + 0.5)
            py = int(60 + 2.0 * t_val + 0.5 * (-0.5) * t_val * t_val + 0.5)
            self.assertGreater(fake_stack.sci[t_idx][py, px], 50.0)
            self.assertLess(fake_stack.sci[t_idx][py, px], 100.0)

            # Far away from the object, the signal should be zero.
            self.assertAlmostEqual(fake_stack.sci[t_idx][30, 40], 0.0)

    def test_create_fake_data_set(self):
        times = create_fake_times(10)
        ds = FakeDataSet(256, 128, times)
        self.assertEqual(ds.stack_py.num_times, 10)

        last_time = -1.0
        for i in range(ds.stack_py.num_times):
            self.assertEqual(ds.stack_py.sci[i].shape, (128, 256))
            self.assertEqual(ds.stack_py.var[i].shape, (128, 256))

            t = ds.stack_py.times[i]
            self.assertGreater(t, last_time)
            last_time = t

    def test_create_fake_data_set_parameters(self):
        """Test that we can create a FakeDataSet with specific noise parameters."""
        times = create_fake_times(10)
        ds = FakeDataSet(
            256,
            256,
            times,
            mask_fraction=0.3,
            noise_level=0.5,
            use_seed=105,
        )
        self.assertEqual(ds.stack_py.num_times, 10)

        for i in range(ds.stack_py.num_times):
            masked = ds.stack_py.get_mask(i)
            self.assertAlmostEqual(np.sum(masked) / ds.stack_py.npixels, 0.3, delta=0.1)

            # Check that we get the expected mean and std of science.
            self.assertAlmostEqual(np.mean(ds.stack_py.sci[i][~masked]), 0.0, delta=0.05)
            self.assertAlmostEqual(np.std(ds.stack_py.sci[i][~masked]), 0.5, delta=0.05)
            self.assertTrue(np.all(ds.stack_py.var[i][~masked] == 0.25))

    def test_regenerate_images(self):
        times = create_fake_times(10)
        ds = FakeDataSet(
            256,
            256,
            times,
            mask_fraction=0.3,
            noise_level=0.5,
            use_seed=105,
        )
        ds.insert_random_object(100)
        self.assertEqual(len(ds.trajectories), 1)
        self.assertEqual(ds.stack_py.num_times, 10)

        old_stack = ds.stack_py.copy()

        # Reset the fake images.
        ds.reset()
        self.assertEqual(len(ds.trajectories), 0)

        self.assertEqual(ds.stack_py.num_times, old_stack.num_times)
        self.assertEqual(ds.stack_py.width, old_stack.width)
        self.assertEqual(ds.stack_py.height, old_stack.height)

        for i in range(ds.stack_py.num_times):
            # Check that the new images are different from the old ones.
            self.assertFalse(np.array_equal(ds.stack_py.sci[i], old_stack.sci[i], equal_nan=True))
            self.assertFalse(np.array_equal(ds.stack_py.var[i], old_stack.var[i], equal_nan=True))

    def test_fake_data_set_insert_artifacts(self):
        """Test that we can insert artifacts into a FakeDataSet."""
        width = 200
        height = 300
        times = create_fake_times(10)
        ds = FakeDataSet(
            width,
            height,
            times,
            mask_fraction=0.0,  # No masking
            noise_level=0.1,  # Low noise level
            use_seed=105,
        )

        # Check that everything is basically noise.
        self.assertEqual(ds.stack_py.num_times, 10)
        for i in range(ds.stack_py.num_times):
            self.assertEqual(np.count_nonzero(ds.stack_py.sci[i] > 2.0), 0)

        # Insert a bunch of artifacts and test that they are inserted correctly.
        ds.insert_random_artifacts(0.1, 20.0, 0.1)
        for i in range(ds.stack_py.num_times):
            artifacts = ds.stack_py.sci[i] > 2.0

            self.assertAlmostEqual(np.sum(artifacts) / (width * height), 0.1, delta=0.1)
            self.assertAlmostEqual(np.mean(ds.stack_py.sci[i][artifacts]), 20.0, delta=0.2)
            self.assertAlmostEqual(np.mean(ds.stack_py.sci[i][~artifacts]), 0.0, delta=0.2)

    def test_insert_object(self):
        times = create_fake_times(5, 57130.2, 3, 0.01, 1)
        ds = FakeDataSet(128, 128, times, use_seed=101)
        self.assertEqual(ds.stack_py.num_times, 5)
        self.assertEqual(len(ds.trajectories), 0)

        # Create and insert a random object.
        trj = ds.insert_random_object(500)
        self.assertEqual(len(ds.trajectories), 1)

        # Check the object was inserted correctly.
        t0 = ds.stack_py.times[0]
        for i in range(ds.stack_py.num_times):
            dt = ds.stack_py.times[i] - t0
            px = trj.get_x_index(dt)
            py = trj.get_y_index(dt)

            # Check the trajectory stays in the image.
            self.assertGreaterEqual(px, 0)
            self.assertGreaterEqual(py, 0)
            self.assertLess(px, 256)
            self.assertLess(py, 256)

            # Check that there is a bright spot at the predicted position.
            pix_val = ds.stack_py.sci[i][py, px]
            self.assertGreaterEqual(pix_val, 50.0)

    def test_trajectory_is_within_bounds(self):
        # Create a dataset with small images.
        width = 30
        height = 40
        num_times = 3
        times = create_fake_times(num_times, 57130.2, 1)
        ds = FakeDataSet(width, height, times, use_seed=101)

        self.assertTrue(ds.trajectory_is_within_bounds(Trajectory(x=0, y=0, vx=1.0, vy=2.0)))
        self.assertTrue(ds.trajectory_is_within_bounds(Trajectory(x=10, y=15, vx=1.0, vy=2.0)))
        self.assertTrue(ds.trajectory_is_within_bounds(Trajectory(x=10, y=15, vx=-1.0, vy=2.0)))
        self.assertFalse(ds.trajectory_is_within_bounds(Trajectory(x=0, y=0, vx=-1.0, vy=1.0)))
        self.assertFalse(ds.trajectory_is_within_bounds(Trajectory(x=0, y=0, vx=1.0, vy=-1.0)))
        self.assertFalse(ds.trajectory_is_within_bounds(Trajectory(x=width - 1, y=0, vx=1.0, vy=1.0)))
        self.assertFalse(ds.trajectory_is_within_bounds(Trajectory(x=0, y=height - 1, vx=1.0, vy=1.0)))

    def test_insert_object_given_vel(self):
        # Create very small images, so we need to be careful with the velocities.
        width = 30
        height = 40
        num_times = 3
        times = create_fake_times(num_times, 57130.2, 1)
        ds = FakeDataSet(width, height, times, use_seed=101)
        self.assertEqual(len(ds.trajectories), 0)

        # Create and insert a random object, check that it has the given velocity.
        trj = ds.insert_random_object(500, vx=1.0, vy=2.0)
        self.assertEqual(len(ds.trajectories), 1)
        self.assertEqual(trj.vx, 1.0)
        self.assertEqual(trj.vy, 2.0)

        vx = [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]
        vy = [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]
        for _ in range(100):
            trj = ds.insert_random_object(500, vx=vx, vy=vy)
            self.assertTrue(trj.vx in vx)
            self.assertTrue(trj.vy in vy)

            # Check that the object is in view at the first time.
            self.assertGreaterEqual(trj.x, 0)
            self.assertGreaterEqual(trj.y, 0)
            self.assertLess(trj.x, width)
            self.assertLess(trj.y, height)

            # Check that the object is still in view at the last time.
            xe = int(trj.x + 2.0 * trj.vx)
            ye = int(trj.y + 2.0 * trj.vy)
            self.assertGreaterEqual(xe, 0)
            self.assertGreaterEqual(ye, 0)
            self.assertLess(xe, width)
            self.assertLess(ye, height)

    def test_insert_object_given_generator(self):
        # Create very small images, so we need to be careful with the velocities.
        width = 30
        height = 40
        num_times = 3
        times = create_fake_times(num_times, 57130.2, 1)
        ds = FakeDataSet(width, height, times, use_seed=101)
        self.assertEqual(len(ds.trajectories), 0)

        trj_generator = VelocityGridSearch(11, 0.0, 20.0, 11, -10.0, 10.0)
        vx = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        vy = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        trjs = ds.insert_random_objects_from_generator(100, trj_generator, 100)
        for trj in trjs:
            self.assertTrue(trj.vx in vx)
            self.assertTrue(trj.vy in vy)

            # Check that the object is in view at the first time.
            self.assertGreaterEqual(trj.x, 0)
            self.assertGreaterEqual(trj.y, 0)
            self.assertLess(trj.x, width)
            self.assertLess(trj.y, height)

            # Check that the object is still in view at the last time.
            xe = int(trj.x + 2.0 * trj.vx)
            ye = int(trj.y + 2.0 * trj.vy)
            self.assertGreaterEqual(xe, 0)
            self.assertGreaterEqual(ye, 0)
            self.assertLess(xe, width)
            self.assertLess(ye, height)

    def test_save_work_unit(self):
        num_images = 25
        ds = FakeDataSet(15, 10, create_fake_times(num_images))
        ds.set_wcs(make_fake_wcs(10.0, 15.0, 15, 10))

        with tempfile.TemporaryDirectory() as dir_name:
            file_name = os.path.join(dir_name, "fake_work_unit.fits")
            ds.save_fake_data_to_work_unit(file_name)
            self.assertTrue(Path(file_name).exists())

            work2 = WorkUnit.from_fits(file_name, show_progress=False)
            self.assertEqual(work2.im_stack.num_times, num_images)
            for i in range(num_images):
                li = work2.im_stack.get_single_image(i)
                self.assertEqual(li.width, 15)
                self.assertEqual(li.height, 10)

    def test_make_results(self):
        num_images = 25
        times = create_fake_times(num_images, 57130.2, 3, 0.01, 1)
        ds = FakeDataSet(150, 200, times)

        # Insert three random objects.
        ds.insert_random_object(500)
        ds.insert_random_object(200)
        ds.insert_random_object(10)  # very dim

        # Create results with all stamps and PSI/PHI images.
        results = ds.make_results(stamp_radius=5)
        self.assertEqual(len(results), 3)
        self.assertTrue("psi_curve" in results.colnames)
        self.assertEqual(results["psi_curve"].shape, (3, num_images))

        self.assertTrue("phi_curve" in results.colnames)
        self.assertEqual(results["phi_curve"].shape, (3, num_images))

        self.assertTrue("coadd_mean" in results.colnames)
        self.assertEqual(results["coadd_mean"].shape, (3, 11, 11))

        self.assertTrue("coadd_median" in results.colnames)
        self.assertEqual(results["coadd_median"].shape, (3, 11, 11))

        self.assertTrue("coadd_sum" in results.colnames)
        self.assertEqual(results["coadd_sum"].shape, (3, 11, 11))

        self.assertTrue("all_stamps" in results.colnames)
        self.assertEqual(results["all_stamps"].shape, (3, num_images, 11, 11))


if __name__ == "__main__":
    unittest.main()
