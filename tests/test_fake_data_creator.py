import numpy as np
import os
import tempfile
import unittest

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

    def test_add_fake_object(self):
        img = RawImage(40, 20, 0.0)  # All zero image.
        p = np.full((3, 3), 1.0 / 9.0)  # Equal PSF.
        add_fake_object(img, 5.5, 3.5, 100.0, p)

        for r in range(10):
            for c in range(20):
                pix_val = img.get_pixel(r, c)
                if abs(c - 5) <= 1 and abs(r - 3) <= 1:
                    self.assertAlmostEqual(pix_val, 100.0 / 9.0, delta=0.001)
                else:
                    self.assertEqual(pix_val, 0.0)

        # Add a fake object with no PSF (right on the edge of the image).
        add_fake_object(img, 39, 19, 100.0, None)
        self.assertAlmostEqual(img.get_pixel(19, 39), 100.0)

        # We don't fail, but do nothing, when we try to insert something
        # off the edge of the image.
        add_fake_object(img, 50.1, 10.0, 100.0, None)

    def test_create(self):
        times = create_fake_times(10)
        ds = FakeDataSet(256, 128, times)
        self.assertEqual(ds.stack.img_count(), 10)

        last_time = -1.0
        for i in range(ds.stack.img_count()):
            layered = ds.stack.get_single_image(i)
            self.assertEqual(layered.get_width(), 256)
            self.assertEqual(layered.get_height(), 128)

            t = layered.get_obstime()
            self.assertGreater(t, last_time)
            last_time = t

    def test_create_empty_times(self):
        ds = FakeDataSet(256, 128, [])
        self.assertEqual(ds.stack.img_count(), 0)

    def test_insert_object(self):
        times = create_fake_times(5, 57130.2, 3, 0.01, 1)
        ds = FakeDataSet(128, 128, times, use_seed=True)
        self.assertEqual(ds.stack.img_count(), 5)
        self.assertEqual(len(ds.trajectories), 0)

        # Create and insert a random object.
        trj = ds.insert_random_object(500)
        self.assertEqual(len(ds.trajectories), 1)

        # Check the object was inserted correctly.
        t0 = ds.stack.get_single_image(0).get_obstime()
        for i in range(ds.stack.img_count()):
            dt = ds.stack.get_single_image(i).get_obstime() - t0
            px = trj.get_x_index(dt)
            py = trj.get_y_index(dt)

            # Check the trajectory stays in the image.
            self.assertGreaterEqual(px, 0)
            self.assertGreaterEqual(py, 0)
            self.assertLess(px, 256)
            self.assertLess(py, 256)

            # Check that there is a bright spot at the predicted position.
            pix_val = ds.stack.get_single_image(i).get_science().get_pixel(py, px)
            self.assertGreaterEqual(pix_val, 50.0)

    def test_save_work_unit(self):
        num_images = 25
        ds = FakeDataSet(15, 10, create_fake_times(num_images))
        ds.set_wcs(make_fake_wcs(10.0, 15.0, 15, 10))

        with tempfile.TemporaryDirectory() as dir_name:
            file_name = os.path.join(dir_name, "fake_work_unit.fits")
            ds.save_fake_data_to_work_unit(file_name)
            self.assertTrue(Path(file_name).exists())

            work2 = WorkUnit.from_fits(file_name)
            self.assertEqual(work2.im_stack.img_count(), num_images)
            for i in range(num_images):
                li = work2.im_stack.get_single_image(i)
                self.assertEqual(li.get_width(), 15)
                self.assertEqual(li.get_height(), 10)


if __name__ == "__main__":
    unittest.main()
