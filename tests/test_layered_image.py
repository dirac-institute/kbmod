import math
import numpy as np
import unittest

from astropy.io import fits

from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import add_fake_object, make_fake_layered_image
from kbmod.search import *


class test_LayeredImage(unittest.TestCase):
    def setUp(self):
        self.p = PSF.make_gaussian_kernel(1.0)
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
        science = self.image.get_science()
        variance = self.image.get_variance()
        mask = self.image.get_mask()
        science_arr = self.image.get_science_array()
        variance_arr = self.image.get_variance_array()
        mask_arr = self.image.get_mask_array()

        for y in range(self.image.height):
            for x in range(self.image.width):
                self.assertEqual(mask.get_pixel(y, x), 0)
                self.assertEqual(variance.get_pixel(y, x), 4.0)

                # These will be potentially flakey due to the random
                # creation (but with very low probability).
                self.assertGreaterEqual(science.get_pixel(y, x), -100.0)
                self.assertLessEqual(science.get_pixel(y, x), 100.0)

                # Check the arrays.
                self.assertEqual(mask_arr[y, x], 0)
                self.assertEqual(variance_arr[y, x], 4.0)
                self.assertAlmostEqual(science.get_pixel(y, x), science_arr[y, x])

        # Check that setting a masked bit does not propagate to the science
        # and variance layers until apply mask is called..
        mask.set_pixel(5, 6, 1)
        self.assertTrue(pixel_value_valid(science.get_pixel(5, 6)))
        self.assertTrue(pixel_value_valid(variance.get_pixel(5, 6)))

    def test_create_from_layers(self):
        sci = RawImage(30, 40)
        for y in range(40):
            for x in range(30):
                sci.set_pixel(y, x, x + 40.0 * y)

        var = RawImage(30, 40)
        var.set_all(1.0)

        mask = RawImage(30, 40)
        mask.set_all(0.0)
        mask.set_pixel(10, 12, 1)

        # Create the layered image.
        img2 = LayeredImage(sci, var, mask, PSF.make_gaussian_kernel(2.0))
        self.assertEqual(img2.width, 30)
        self.assertEqual(img2.height, 40)
        self.assertEqual(img2.get_npixels(), 30.0 * 40.0)
        self.assertEqual(img2.time, -1.0)  # No time given

        # Check the layers.
        science = img2.get_science()
        variance = img2.get_variance()
        mask2 = img2.get_mask()
        for y in range(img2.height):
            for x in range(img2.width):
                if x == 12 and y == 10:
                    # The masked pixel should have no data.
                    self.assertEqual(mask2.get_pixel(y, x), 1)
                else:
                    self.assertEqual(mask2.get_pixel(y, x), 0)

                # The individual layers do not have the masking until it is applied.
                self.assertEqual(variance.get_pixel(y, x), 1.0)
                self.assertAlmostEqual(science.get_pixel(y, x), x + 40.0 * y)

    def test_convolve_psf(self):
        sci0 = self.image.get_science()
        var0 = self.image.get_variance()
        msk0 = self.image.get_mask()

        # Create a copy of the image.
        img_b = LayeredImage(sci0, var0, msk0, self.p)

        # A no-op PSF does not change the image.
        img_b.convolve_given_psf(np.array([[1.0]]))
        sci1 = img_b.get_science()
        var1 = img_b.get_variance()
        for y in range(img_b.height):
            for x in range(img_b.width):
                self.assertAlmostEqual(sci0.get_pixel(y, x), sci1.get_pixel(y, x))
                self.assertAlmostEqual(var0.get_pixel(y, x), var1.get_pixel(y, x))

        # The default PSF (stdev=1.0) DOES have the image.
        img_b.convolve_psf()
        sci1 = img_b.get_science()
        var1 = img_b.get_variance()
        for y in range(img_b.height):
            for x in range(img_b.width):
                self.assertNotAlmostEqual(sci0.get_pixel(y, x), sci1.get_pixel(y, x))
                self.assertNotAlmostEqual(var0.get_pixel(y, x), var1.get_pixel(y, x))

    def test_overwrite_PSF(self):
        p1 = self.image.get_psf()
        self.assertEqual(p1.shape, (7, 7))

        # Get the science pixel with the original PSF blurring.
        science_org = self.image.get_science()
        add_fake_object(self.image, 50, 50, 500.0, p1)
        science_pixel_psf1 = self.image.get_science().get_pixel(50, 50)

        # Change the PSF to a no-op.
        self.image.set_psf(np.array([[1.0]]))

        # Check that we retrieve the correct PSF.
        p2 = self.image.get_psf()
        self.assertEqual(p2.shape, (1, 1))

        # Check that the science pixel with the new PSF blurring is
        # larger (because the PSF is tighter).
        self.image.set_science(science_org)
        add_fake_object(self.image, 50, 50, 500.0, p2)
        science_pixel_psf2 = self.image.get_science().get_pixel(50, 50)
        self.assertLess(science_pixel_psf1, science_pixel_psf2)

    def test_mask_pixel(self):
        self.image.mask_pixel(10, 15)
        self.image.mask_pixel(22, 23)
        for y in range(self.image.height):
            for x in range(self.image.width):
                pix_val = self.image.get_science().get_pixel(y, x)
                expected = not ((x == 15 and y == 10) or (x == 23 and y == 22))
                self.assertEqual(pixel_value_valid(pix_val), expected)

    def test_binarize_mask(self):
        # Mask out a range of pixels.
        mask = self.image.get_mask()
        for x in range(9):
            mask.set_pixel(10, x, x)

        # Only keep the mask for pixels with flags at
        # bit positions 0 and 2 (1 + 4 = 5).
        self.image.binarize_mask(5)
        self.assertEqual(mask.get_pixel(10, 0), 0)
        self.assertEqual(mask.get_pixel(10, 1), 1)
        self.assertEqual(mask.get_pixel(10, 2), 0)
        self.assertEqual(mask.get_pixel(10, 3), 1)
        self.assertEqual(mask.get_pixel(10, 4), 1)
        self.assertEqual(mask.get_pixel(10, 5), 1)
        self.assertEqual(mask.get_pixel(10, 6), 1)
        self.assertEqual(mask.get_pixel(10, 7), 1)
        self.assertEqual(mask.get_pixel(10, 8), 0)

    def test_apply_mask(self):
        # Nothing is initially masked.
        science = self.image.get_science()
        for y in range(self.image.height):
            for x in range(self.image.width):
                self.assertTrue(science.pixel_has_data(y, x))

        # Mask out three pixels.
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 2)
        mask.set_pixel(10, 13, 3)

        # Apply the mask flags to only (10, 11) and (10, 13)
        self.image.apply_mask(1)

        science = self.image.get_science()
        for y in range(self.image.height):
            for x in range(self.image.width):
                if y == 10 and (x == 11 or x == 13):
                    self.assertFalse(science.pixel_has_data(y, x))
                else:
                    self.assertTrue(science.pixel_has_data(y, x))

    def test_psi_and_phi_image(self):
        p = PSF.make_gaussian_kernel(0.00000001)  # A point function.
        img = make_fake_layered_image(6, 5, 2.0, 4.0, 10.0, p)

        # Create fake science and variance images.
        sci = img.get_science()
        var = img.get_variance()
        for y in range(5):
            for x in range(6):
                sci.set_pixel(y, x, float(x))
                var.set_pixel(y, x, float(y + 1))

        # Mask a single pixel, set another to variance of zero,
        # and mark two as NaN.
        sci.mask_pixel(3, 1)
        var.mask_pixel(3, 1)
        var.set_pixel(3, 2, 0.0)
        var.set_pixel(3, 0, np.nan)
        sci.set_pixel(3, 3, math.nan)
        sci.set_pixel(3, 4, np.nan)

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
