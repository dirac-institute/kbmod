import numpy as np
import unittest

from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.search import ImageStack, LayeredImage, PSF, StampCreator, Trajectory


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

    def test_get_variance_weighted_stamp(self):
        sci1 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.single)
        var1 = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.single)
        msk1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.single)
        layer1 = LayeredImage(sci1, var1, msk1, PSF(1e-12), 0.0)
        layer1.apply_mask(0xFFFFFF)

        sci2 = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.single)
        var2 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.single)
        msk2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.single)
        layer2 = LayeredImage(sci2, var2, msk2, PSF(1e-12), 0.0)
        layer2.apply_mask(0xFFFFFF)

        stack = ImageStack([layer1, layer2])

        # Unmoving point in the center. Result should be (1.0 / 1.0 + 2.0 / 0.5) / (1.0 / 1.0 + 1.0 / 0.5)
        stamp = self.stamp_creator.get_variance_weighted_stamp(stack, Trajectory(1, 1, 0.0, 0.0), 0)
        self.assertEqual(stamp.image.shape, (1, 1))
        self.assertAlmostEqual(stamp.get_pixel(0, 0), 5.0 / 3.0)

        # Unmoving point in the top corner. Should ignore the point in the second image.
        stamp = self.stamp_creator.get_variance_weighted_stamp(stack, Trajectory(0, 0, 0.0, 0.0), 0)
        self.assertEqual(stamp.image.shape, (1, 1))
        self.assertAlmostEqual(stamp.get_pixel(0, 0), 1.0)

        # Unmoving point in the bottom corner. Should ignore the point in the first image.
        stamp = self.stamp_creator.get_variance_weighted_stamp(stack, Trajectory(2, 2, 0.0, 0.0), 0)
        self.assertEqual(stamp.image.shape, (1, 1))
        self.assertAlmostEqual(stamp.get_pixel(0, 0), 2.0)


if __name__ == "__main__":
    unittest.main()
