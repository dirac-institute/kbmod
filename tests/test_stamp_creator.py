import numpy as np
import unittest

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.search import StampCreator, Trajectory


class test_stamp_creator(unittest.TestCase):
    def setUp(self):
        # Create a fake data set to use in the tests.
        self.image_count = 10
        self.fake_times = create_fake_times(self.image_count, 57130.2, 1, 0.01, 1)
        self.ds = FakeDataSet(
            25,  # width
            35,  # height
            self.fake_times,  # time stamps
            1.0,  # noise level
            0.5,  # psf value
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        self.trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        self.ds.insert_object(self.trj)

        # Make a StampCreator.
        self.stamp_creator = StampCreator()

    def test_create_stamps(self):
        stamps = self.stamp_creator.create_stamps(self.ds.stack, self.trj, 1, True, [])
        self.assertEqual(len(stamps), self.image_count)
        for i in range(self.image_count):
            self.assertEqual(stamps[i].image.shape, (3, 3))

            pix_val = self.ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(pix_val):
                self.assertTrue(np.isnan(stamps[i].get_pixel(1, 1)))
            else:
                self.assertAlmostEqual(pix_val, stamps[i].get_pixel(1, 1))

        # Check that we can set use_indices to produce only some stamps.
        use_times = [False, True, False, True, True, False, False, False, True, False]
        stamps = self.stamp_creator.create_stamps(self.ds.stack, self.trj, 1, True, use_times)
        self.assertEqual(len(stamps), np.count_nonzero(use_times))

        stamp_count = 0
        for i in range(self.image_count):
            if use_times[i]:
                self.assertEqual(stamps[stamp_count].image.shape, (3, 3))

                pix_val = self.ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
                if np.isnan(pix_val):
                    self.assertTrue(np.isnan(stamps[stamp_count].get_pixel(1, 1)))
                else:
                    self.assertAlmostEqual(pix_val, stamps[stamp_count].get_pixel(1, 1))

                stamp_count += 1

    def test_create_variance_stamps(self):
        test_trj = Trajectory(8, 7, 1.0, 2.0)
        stamps = self.stamp_creator.create_variance_stamps(self.ds.stack, self.trj, 1)
        self.assertEqual(len(stamps), self.image_count)
        for i in range(self.image_count):
            self.assertEqual(stamps[i].image.shape, (3, 3))

            pix_val = self.ds.stack.get_single_image(i).get_variance().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(pix_val):
                self.assertTrue(np.isnan(stamps[i].get_pixel(1, 1)))
            else:
                self.assertAlmostEqual(pix_val, stamps[i].get_pixel(1, 1))


if __name__ == "__main__":
    unittest.main()
