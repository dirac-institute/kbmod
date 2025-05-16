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

    def test_create(self):
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
