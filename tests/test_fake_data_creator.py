import tempfile
import unittest

from kbmod.fake_data_creator import *
from kbmod.file_utils import *
from kbmod.search import *


class test_fake_image_creator(unittest.TestCase):
    def test_create(self):
        ds = FakeDataSet(128, 256, 10)
        self.assertEqual(ds.stack.img_count(), 10)

        last_time = -1.0
        for i in range(ds.stack.img_count()):
            layered = ds.stack.get_single_image(i)
            self.assertEqual(layered.get_width(), 128)
            self.assertEqual(layered.get_height(), 256)

            t = layered.get_obstime()
            self.assertGreater(t, last_time)
            last_time = t

    def test_insert_object(self):
        ds = FakeDataSet(128, 128, 5, use_seed=True)
        self.assertEqual(ds.stack.img_count(), 5)
        self.assertEqual(len(ds.trajectories), 0)

        # Create and insert a random object.
        trj = ds.insert_random_object(500)
        self.assertEqual(len(ds.trajectories), 1)

        # Check the object was inserted correctly.
        t0 = ds.stack.get_single_image(0).get_obstime()
        for i in range(ds.stack.img_count()):
            dt = ds.stack.get_single_image(i).get_obstime() - t0
            px = int(trj.x + dt * trj.vx + 0.5)
            py = int(trj.y + dt * trj.vy + 0.5)

            # Check the trajectory stays in the image.
            self.assertGreaterEqual(px, 0)
            self.assertGreaterEqual(py, 0)
            self.assertLess(px, 256)
            self.assertLess(py, 256)

            # Check that there is a bright spot at the predicted position.
            pix_val = ds.stack.get_single_image(i).get_science().get_pixel(px, py)
            self.assertGreaterEqual(pix_val, 50.0)

    def test_save_and_clean(self):
        num_images = 7
        ds = FakeDataSet(64, 64, num_images)

        with tempfile.TemporaryDirectory() as dir_name:
            # Get all the file names.
            filenames = []
            for i in range(num_images):
                image_name = ds.stack.get_single_image(i).get_name()
                filenames.append(f"{dir_name}/{image_name}.fits")

            # Check no data exists yet.
            for name in filenames:
                self.assertFalse(Path(name).exists())

            # Save the data and check the data now exists.
            ds.save_fake_data(dir_name)
            for name in filenames:
                self.assertTrue(Path(name).exists())

            # Clean the data and check the data no longer exists.
            ds.delete_fake_data(dir_name)
            for name in filenames:
                self.assertFalse(Path(name).exists())

    def test_save_times(self):
        num_images = 50
        ds = FakeDataSet(4, 4, num_images)

        with tempfile.TemporaryDirectory() as dir_name:
            file_name = f"{dir_name}/times.dat"
            ds.save_time_file(file_name)
            self.assertTrue(Path(file_name).exists())

            time_load = FileUtils.load_time_dictionary(file_name)
            self.assertEqual(len(time_load), 50)


if __name__ == "__main__":
    unittest.main()
