import math
import numpy as np
import os
import tempfile
import unittest

from astropy.io import fits

from kbmod.fake_data.fake_data_creator import add_fake_object, make_fake_layered_image
from kbmod.search import *


class test_LayeredImage(unittest.TestCase):
    def setUp(self):
        self.p = PSF(1.0)
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
        self.assertEqual(self.image.get_width(), 60)
        self.assertEqual(self.image.get_height(), 80)
        self.assertEqual(self.image.get_npixels(), 80 * 60)
        self.assertEqual(self.image.get_obstime(), 10.0)

        # Create a fake LayeredImage.
        science = self.image.get_science()
        variance = self.image.get_variance()
        mask = self.image.get_mask()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                self.assertEqual(mask.get_pixel(y, x), 0)
                self.assertEqual(variance.get_pixel(y, x), 4.0)

                # These will be potentially flakey due to the random
                # creation (but with very low probability).
                self.assertGreaterEqual(science.get_pixel(y, x), -100.0)
                self.assertLessEqual(science.get_pixel(y, x), 100.0)

                # Check direct lookup of pixel values matches the RawImage lookup.
                self.assertEqual(science.get_pixel(y, x), self.image.get_science_pixel(y, x))
                self.assertEqual(variance.get_pixel(y, x), self.image.get_variance_pixel(y, x))

        # Check that the LayeredImage pixel lookups work with a masked pixel.
        # But the the mask was not applied yet to the images themselves.
        mask.set_pixel(5, 6, 1)
        self.assertTrue(pixel_value_valid(science.get_pixel(5, 6)))
        self.assertTrue(pixel_value_valid(variance.get_pixel(5, 6)))
        self.assertFalse(pixel_value_valid(self.image.get_science_pixel(5, 6)))
        self.assertFalse(pixel_value_valid(self.image.get_variance_pixel(5, 6)))

        # Test that out of bounds pixel lookups are handled correctly.
        self.assertFalse(pixel_value_valid(self.image.get_science_pixel(-1, 1)))
        self.assertFalse(pixel_value_valid(self.image.get_science_pixel(1, -1)))
        self.assertFalse(pixel_value_valid(self.image.get_science_pixel(self.image.get_height() + 1, 1)))
        self.assertFalse(pixel_value_valid(self.image.get_science_pixel(1, self.image.get_width() + 1)))

        self.assertFalse(pixel_value_valid(self.image.get_variance_pixel(-1, 1)))
        self.assertFalse(pixel_value_valid(self.image.get_variance_pixel(1, -1)))
        self.assertFalse(pixel_value_valid(self.image.get_variance_pixel(self.image.get_height() + 1, 1)))
        self.assertFalse(pixel_value_valid(self.image.get_variance_pixel(1, self.image.get_width() + 1)))

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
        img2 = LayeredImage(sci, var, mask, PSF(2.0))
        self.assertEqual(img2.get_width(), 30)
        self.assertEqual(img2.get_height(), 40)
        self.assertEqual(img2.get_npixels(), 30.0 * 40.0)
        self.assertEqual(img2.get_obstime(), -1.0)  # No time given

        # Test the bounds checking.
        self.assertTrue(img2.contains(0, 0))
        self.assertTrue(img2.contains(39, 29))
        self.assertFalse(img2.contains(39, 30))
        self.assertFalse(img2.contains(40, 15))
        self.assertFalse(img2.contains(15, -1))
        self.assertFalse(img2.contains(-1, 0))

        # Check the layers.
        science = img2.get_science()
        variance = img2.get_variance()
        mask2 = img2.get_mask()
        for y in range(img2.get_height()):
            for x in range(img2.get_width()):
                if x == 12 and y == 10:
                    # The masked pixel should have no data.
                    self.assertEqual(mask2.get_pixel(y, x), 1)
                    self.assertFalse(pixel_value_valid(img2.get_science_pixel(y, x)))
                    self.assertFalse(pixel_value_valid(img2.get_variance_pixel(y, x)))
                else:
                    self.assertEqual(mask2.get_pixel(y, x), 0)
                    self.assertAlmostEqual(img2.get_science_pixel(y, x), x + 40.0 * y)
                    self.assertAlmostEqual(img2.get_variance_pixel(y, x), 1.0)

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
        img_b.convolve_given_psf(PSF())
        sci1 = img_b.get_science()
        var1 = img_b.get_variance()
        for y in range(img_b.get_height()):
            for x in range(img_b.get_width()):
                self.assertAlmostEqual(sci0.get_pixel(y, x), sci1.get_pixel(y, x))
                self.assertAlmostEqual(var0.get_pixel(y, x), var1.get_pixel(y, x))

        # The default PSF (stdev=1.0) DOES have the image.
        img_b.convolve_psf()
        sci1 = img_b.get_science()
        var1 = img_b.get_variance()
        for y in range(img_b.get_height()):
            for x in range(img_b.get_width()):
                self.assertNotAlmostEqual(sci0.get_pixel(y, x), sci1.get_pixel(y, x))
                self.assertNotAlmostEqual(var0.get_pixel(y, x), var1.get_pixel(y, x))

    def test_overwrite_PSF(self):
        p1 = self.image.get_psf()
        self.assertEqual(p1.get_size(), 25)
        self.assertEqual(p1.get_dim(), 5)
        self.assertEqual(p1.get_radius(), 2)

        # Get the science pixel with the original PSF blurring.
        science_org = self.image.get_science()
        add_fake_object(self.image, 50, 50, 500.0, p1)
        science_pixel_psf1 = self.image.get_science().get_pixel(50, 50)

        # Change the PSF to a no-op.
        self.image.set_psf(PSF())

        # Check that we retrieve the correct PSF.
        p2 = self.image.get_psf()
        self.assertEqual(p2.get_size(), 1)
        self.assertEqual(p2.get_dim(), 1)
        self.assertEqual(p2.get_radius(), 0)

        # Check that the science pixel with the new PSF blurring is
        # larger (because the PSF is tighter).
        self.image.set_science(science_org)
        add_fake_object(self.image, 50, 50, 500.0, p2)
        science_pixel_psf2 = self.image.get_science().get_pixel(50, 50)
        self.assertLess(science_pixel_psf1, science_pixel_psf2)

    def test_mask_pixel(self):
        self.image.mask_pixel(10, 15)
        self.image.mask_pixel(22, 23)
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                pix_val = self.image.get_science_pixel(y, x)
                expected = not ((x == 15 and y == 10) or (x == 23 and y == 22))
                self.assertEqual(pixel_value_valid(pix_val), expected)
                self.assertEqual(self.image.science_pixel_has_data(y, x), expected)

    def test_compute_fraction_masked(self):
        total_pixels = self.width * self.height

        # Mask 50 pixels
        for y in range(0, 10):
            for x in range(0, 5):
                self.image.mask_pixel(y, x)
        self.assertAlmostEqual(self.image.compute_fraction_masked(), 50.0 / total_pixels)

        # Mask another 25 pixels.
        for x in range(3, 28):
            self.image.mask_pixel(12, x)
        self.assertAlmostEqual(self.image.compute_fraction_masked(), 75.0 / total_pixels)

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
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                self.assertTrue(science.pixel_has_data(y, x))

        # Mask out three pixels.
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 2)
        mask.set_pixel(10, 13, 3)

        # Apply the mask flags to only (10, 11) and (10, 13)
        self.image.apply_mask(1)

        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                if y == 10 and (x == 11 or x == 13):
                    self.assertFalse(science.pixel_has_data(y, x))
                else:
                    self.assertTrue(science.pixel_has_data(y, x))

    def test_psi_and_phi_image(self):
        p = PSF(0.00000001)  # A point function.
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
        self.assertEqual(psi.width, 6)
        self.assertEqual(psi.height, 5)

        phi = img.generate_phi_image()
        self.assertEqual(phi.width, 6)
        self.assertEqual(phi.height, 5)

        for y in range(5):
            for x in range(6):
                psi_has_data = y != 3 or x > 4
                self.assertEqual(psi.pixel_has_data(y, x), psi_has_data)
                if psi_has_data:
                    self.assertAlmostEqual(psi.get_pixel(y, x), x / (y + 1), delta=1e-5)
                else:
                    self.assertFalse(pixel_value_valid(psi.get_pixel(y, x)))

                phi_has_data = y != 3 or x > 2
                self.assertEqual(phi.pixel_has_data(y, x), phi_has_data)
                if phi_has_data:
                    self.assertAlmostEqual(phi.get_pixel(y, x), 1.0 / (y + 1), delta=1e-5)
                else:
                    self.assertFalse(pixel_value_valid(phi.get_pixel(y, x)))

    def test_subtract_template(self):
        sci = self.image.get_science()
        sci.mask_pixel(7, 10)
        sci.mask_pixel(7, 11)
        sci.set_pixel(7, 12, math.nan)
        sci.set_pixel(7, 13, np.nan)
        old_sci = RawImage(sci.image.copy())  # Make a copy.

        template = RawImage(self.image.get_width(), self.image.get_height())
        template.set_all(0.0)
        for h in range(sci.height):
            for w in range(4, sci.width):
                template.set_pixel(h, w, 0.01 * h)
        self.image.sub_template(template)

        for y in range(sci.height):
            for x in range(sci.width):
                if y == 7 and (x >= 10 and x <= 13):
                    self.assertFalse(sci.pixel_has_data(y, x))
                elif x < 4:
                    val1 = old_sci.get_pixel(y, x)
                    val2 = sci.get_pixel(y, x)
                    self.assertEqual(val1, val2)
                else:
                    val1 = old_sci.get_pixel(y, x) - 0.01 * y
                    val2 = sci.get_pixel(y, x)
                    self.assertAlmostEqual(val1, val2, delta=1e-5)

        # Test that we fail when the template size does not match.
        template2 = RawImage(self.image.get_width(), self.image.get_height() + 1)
        template2.set_all(0.0)
        self.assertRaises(RuntimeError, self.image.sub_template, template2)


if __name__ == "__main__":
    unittest.main()
