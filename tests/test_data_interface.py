from kbmod.data_interface import Interface
import unittest
from utils.utils_for_tests import get_absolute_data_path
from kbmod.search import *


class test_data_interface(unittest.TestCase):
    def test_file_load_basic(self):
        loader = Interface()
        stack, wcs_list, mjds = loader.load_images(
            get_absolute_data_path("fake_images"),
            None,
            None,
            [0, 157130.2],
            PSF(1.0),
            verbose=False,
        )
        self.assertEqual(stack.img_count(), 4)

        # Check that each image loaded corrected.
        true_times = [57130.2, 57130.21, 57130.22, 57131.2]
        for i in range(stack.img_count()):
            img = stack.get_single_image(i)
            self.assertEqual(img.get_width(), 64)
            self.assertEqual(img.get_height(), 64)
            self.assertAlmostEqual(img.get_obstime(), true_times[i], delta=0.005)
            self.assertAlmostEqual(1.0, img.get_psf().get_std())

    def test_file_load_extra(self):
        p = PSF(1.0)

        loader = Interface()
        stack, wcs_list, mjds = loader.load_images(
            get_absolute_data_path("fake_images"),
            get_absolute_data_path("fake_times.dat"),
            get_absolute_data_path("fake_psfs.dat"),
            [0, 157130.2],
            p,
            verbose=False,
        )
        self.assertEqual(stack.img_count(), 4)

        # Check that each image loaded corrected.
        true_times = [57130.2, 57130.21, 57130.22, 57162.0]
        psfs_std = [1.0, 1.0, 1.3, 1.0]
        for i in range(stack.img_count()):
            img = stack.get_single_image(i)
            self.assertEqual(img.get_width(), 64)
            self.assertEqual(img.get_height(), 64)
            self.assertAlmostEqual(img.get_obstime(), true_times[i], delta=0.005)
            self.assertAlmostEqual(psfs_std[i], img.get_psf().get_std())


if __name__ == "__main__":
    unittest.main()
