import numpy as np
import os
import tempfile
import unittest

from pathlib import Path

from kbmod.fake_data.fake_data_creator import *
from kbmod.search import *
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

    def test_image_stack_add_fake_object(self):
        """Test that we can inset a fake object into an ImageStackPy."""
        num_times = 5
        height = 200
        width = 300

        fake_times = np.arange(num_times)
        sci = np.full((num_times, height, width), 0.0)
        var = np.full((num_times, height, width), 1.0)
        psfs = [PSF.from_gaussian(0.5) for i in range(num_times)]
        fake_stack = ImageStackPy(fake_times, sci, var, psfs=psfs)

        image_stack_add_fake_object(fake_stack, 50, 60, 1.0, 2.0, 100.0)
        for t_idx, t_val in enumerate(fake_times):
            # Check that we receive a signal at the correct location that
            # is non-zero but less than the flux (due to the PSF) at each time step.
            px = int(50 + t_val + 0.5)
            py = int(60 + 2.0 * t_val + 0.5)
            self.assertGreater(fake_stack.sci[t_idx][py, px], 50.0)
            self.assertLess(fake_stack.sci[t_idx][py, px], 100.0)

            # Far away from the object, the signal should be zero.
            self.assertAlmostEqual(fake_stack.sci[t_idx][30, 40], 0.0)

    def test_get_masked_fractions(self):
        """Test that we can compute the fraction of pixels masked at each index."""
        num_times = 20
        height = 20
        width = 30
        stack = ImageStackPy()

        # Append images with an increasing fraction of masked pixels
        for idx in range(num_times):
            sci = np.full((height, width), float(idx))
            var = np.full((height, width), 0.1 * float(idx))
            msk = np.zeros((height, width))
            msk[0:idx, :] = 1
            stack.append_image(idx / 20.0, sci, var, mask=msk)

        fractions = stack.get_masked_fractions()
        self.assertTrue(np.allclose(fractions, np.arange(num_times) / float(num_times)))

        # Filter out images with above a given masked fraction.
        stack.filter_images(fractions < 0.5)
        times_remaining = int(num_times / 2)
        self.assertEqual(stack.num_times, times_remaining)
        self.assertTrue(np.allclose(stack.times, np.arange(times_remaining) / float(num_times)))

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

    def test_insert_object(self):
        times = create_fake_times(5, 57130.2, 3, 0.01, 1)
        ds = FakeDataSet(128, 128, times, use_seed=True)
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


if __name__ == "__main__":
    unittest.main()
