import math
import numpy as np
import unittest

from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import make_fake_layered_image
from kbmod.search import *


class test_LayeredImage(unittest.TestCase):
    def setUp(self):
        self.p = PSF.make_gaussian_kernel(1.0).astype(np.float32)
        self.width = 60
        self.height = 80

        # Create a fake layered image to use.
        self.image = make_fake_layered_image(
            self.width,
            self.height,
            2.0,  # noise_level
            4.0,  # variance
            10.0,  # time = 10.0
            self.p,
        )

    def test_create(self):
        self.assertIsNotNone(self.image)
        self.assertEqual(self.image.width, 60)
        self.assertEqual(self.image.height, 80)
        self.assertEqual(self.image.get_npixels(), 80 * 60)
        self.assertEqual(self.image.time, 10.0)

        # Check the created image.
        science_arr = self.image.sci
        variance_arr = self.image.var
        mask_arr = self.image.mask

        for y in range(self.image.height):
            for x in range(self.image.width):
                self.assertEqual(mask_arr[y, x], 0)
                self.assertEqual(variance_arr[y, x], 4.0)
                self.assertGreater(science_arr[y, x], -200)
                self.assertLess(science_arr[y, x], 200)

        # Check that setting a masked bit does not propagate to the science
        # and variance layers until apply mask is called..
        variance_arr[5, 6] = 1
        self.assertTrue(pixel_value_valid(science_arr[5, 6]))
        self.assertTrue(pixel_value_valid(variance_arr[5, 6]))

    def test_create_from_layers(self):
        sci = np.zeros((40, 30), dtype=np.float32)
        for y in range(40):
            for x in range(30):
                sci[y, x] = x + 40.0 * y
        var = np.full((40, 30), 1.0, dtype=np.float32)

        mask = np.full((40, 30), 0, dtype=np.float32)
        mask[10, 12] = 1

        # Create the layered image. We use a copy of the
        # arrays so that LayeredImage can take ownership.
        img2 = LayeredImage(
            np.copy(sci),
            np.copy(var),
            np.copy(mask),
            PSF.make_gaussian_kernel(2.0).astype(np.float32),
            1.0,
        )

        self.assertEqual(img2.width, 30)
        self.assertEqual(img2.height, 40)
        self.assertEqual(img2.get_npixels(), 30.0 * 40.0)
        self.assertEqual(img2.time, 1.0)

        # Check the layers.
        self.assertTrue(np.allclose(img2.sci, sci))
        self.assertTrue(np.allclose(img2.var, var))
        self.assertTrue(np.allclose(img2.mask, mask))

    def test_convolve_psf(self):
        sci0 = self.image.sci
        var0 = self.image.var
        msk0 = self.image.mask

        # Create a copy of the image.  We use a copy of the
        # arrays so that LayeredImage can take ownership.
        img_b = LayeredImage(
            np.copy(sci0),
            np.copy(var0),
            np.copy(msk0),
            self.p,
            1.0,
        )

        # A no-op PSF does not change the image.
        img_b.convolve_given_psf(np.array([[1.0]]))
        sci1 = img_b.sci
        var1 = img_b.var
        self.assertTrue(np.allclose(sci0, sci1))
        self.assertTrue(np.allclose(var0, var1))

        # The default PSF (stdev=1.0) DOES have the image.
        img_b.convolve_psf()
        sci1 = img_b.sci
        var1 = img_b.var
        self.assertFalse(np.allclose(sci0, sci1))
        self.assertFalse(np.allclose(var0, var1))

    def test_set_PSF(self):
        p1 = self.image.get_psf()
        self.assertEqual(p1.shape, (7, 7))

        # Change the PSF to a no-op.
        self.image.set_psf(np.array([[1.0]]))

        # Check that we retrieve the correct PSF.
        p2 = self.image.get_psf()
        self.assertEqual(p2.shape, (1, 1))

    def test_apply_mask(self):
        # Nothing is initially masked.
        science = self.image.sci
        self.assertEqual(np.count_nonzero(np.isnan(science)), 0)

        # Mask out three pixels.
        mask = self.image.mask
        mask[10, 11] = 1
        mask[10, 12] = 2
        mask[10, 13] = 3

        # Apply the mask flags to only (10, 11) and (10, 13)
        self.image.apply_mask(1)

        science = self.image.sci
        for y in range(self.image.height):
            for x in range(self.image.width):
                if y == 10 and (x == 11 or x == 13):
                    self.assertTrue(np.isnan(science[y, x]))
                else:
                    self.assertFalse(np.isnan(science[y, x]))

    def test_psi_and_phi_image(self):
        p = PSF.make_gaussian_kernel(0.00000001)  # A point function.
        img = make_fake_layered_image(6, 5, 2.0, 4.0, 10.0, p)

        # Create fake science and variance images.
        sci = img.sci
        var = img.var
        for y in range(5):
            for x in range(6):
                sci[y, x] = float(x)
                var[y, x] = float(y + 1)

        # Mask a single pixel, set another to variance of zero,
        # and mark two as NaN.
        sci[3, 1] = np.nan
        var[3, 1] = np.nan
        var[3, 2] = 0.0
        var[3, 0] = np.nan
        sci[3, 3] = math.nan
        sci[3, 4] = np.nan

        # Generate and check psi and phi images.
        psi = img.generate_psi_image()
        self.assertEqual(psi.shape, (5, 6))

        phi = img.generate_phi_image()
        self.assertEqual(phi.shape, (5, 6))

        for y in range(5):
            for x in range(6):
                psi_val = psi[y, x]
                if y != 3 or x > 4:
                    self.assertTrue(np.isfinite(psi_val))
                    self.assertAlmostEqual(psi_val, x / (y + 1), delta=1e-5)
                else:
                    self.assertFalse(np.isfinite(psi_val))

                phi_val = phi[y, x]
                if y != 3 or x > 2:
                    self.assertTrue(np.isfinite(phi_val))
                    self.assertAlmostEqual(phi_val, 1.0 / (y + 1), delta=1e-5)
                else:
                    self.assertFalse(np.isfinite(phi_val))


if __name__ == "__main__":
    unittest.main()
