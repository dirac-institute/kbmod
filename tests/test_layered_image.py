import tempfile
import unittest

from astropy.io import fits

from kbmod.fake_data_creator import add_fake_object
from kbmod.search import *


class test_LayeredImage(unittest.TestCase):
    def setUp(self):
        self.p = PSF(1.0)

        # Create a fake layered image to use.
        self.image = LayeredImage(
            "layered_test",
            80,  # dim_y = 80 pixels,
            60,  # dim_x = 60 pixels,
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
        self.assertEqual(self.image.get_name(), "layered_test")

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

    def test_create_from_layers(self):
        sci = RawImage(30, 40)
        for y in range(30):
            for x in range(40):
                sci.set_pixel(y, x, x + 40.0 * y)

        var = RawImage(30, 40)
        var.set_all(1.0)

        mask = RawImage(30, 40)
        mask.set_all(0.0)

        # Create the layered image.
        img2 = LayeredImage(sci, var, mask, PSF(2.0))
        self.assertEqual(img2.get_width(), 40)
        self.assertEqual(img2.get_height(), 30)
        self.assertEqual(img2.get_npixels(), 30.0 * 40.0)
        self.assertEqual(img2.get_obstime(), -1.0)  # No time given

        # Check the layers.
        science = img2.get_science()
        variance = img2.get_variance()
        mask2 = img2.get_mask()
        for y in range(img2.get_height()):
            for x in range(img2.get_width()):
                self.assertEqual(mask2.get_pixel(y, x), 0)
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

    def test_mask_threshold(self):
        masked_pixels = {}
        threshold = 20.0

        # Add an object brighter than the threshold.
        add_fake_object(self.image, 50, 50, 500.0, self.p)

        # Find all the pixels that should be masked.
        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                value = science.get_pixel(y, x)
                if value > threshold:
                    index = self.image.get_width() * y + x
                    masked_pixels[index] = True

        # Do the masking and confirm we have masked
        # at least 1 pixel.
        self.image.apply_mask_threshold(threshold)
        self.assertGreater(len(masked_pixels), 0)

        # Check that we masked the correct pixels.
        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                index = self.image.get_width() * y+ x
                if index in masked_pixels:
                    self.assertFalse(science.pixel_has_data(y, x))
                else:
                    self.assertTrue(science.pixel_has_data(y, x))

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
        self.image.apply_mask_flags(1, [])

        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                if y == 10 and (x == 11 or x == 13):
                    self.assertFalse(science.pixel_has_data(y, x))
                else:
                    self.assertTrue(science.pixel_has_data(y, x))

    def test_apply_mask_exceptions(self):
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 2)
        mask.set_pixel(10, 13, 3)

        # Apply the mask flags to only (10, 11).
        self.image.apply_mask_flags(1, [1])

        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                if y == 10 and x == 13:
                    self.assertFalse(science.pixel_has_data(y, x))
                else:
                    self.assertTrue(science.pixel_has_data(y, x))

    def test_grow_mask(self):
        mask = self.image.get_mask()
        mask.set_pixel(11, 10, 1)
        mask.set_pixel(12, 10, 1)
        mask.set_pixel(13, 10, 1)
        self.image.apply_mask_flags(1, [])
        self.image.grow_mask(1)

        # Check that the mask has grown to all adjacent pixels.
        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                should_mask = (
                    (x == 10 and y <= 14 and y >= 10)
                    or (x == 9 and y <= 13 and y >= 11)
                    or (x == 11 and y <= 13 and y >= 11)
                )
                self.assertEqual(science.pixel_has_data(y, x), not should_mask)

    def test_grow_mask_mult(self):
        mask = self.image.get_mask()
        mask.set_pixel(11, 10, 1)
        mask.set_pixel(12, 10, 1)
        self.image.apply_mask_flags(1, [])
        self.image.grow_mask(3)

        # Check that the mask has grown to all applicable pixels.
        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                # Check whether the point is a manhattan distance of <= 3 from
                # one of the original masked pixels.
                dx = abs(x - 10)
                dy = min(abs(y - 11), abs(y - 12))
                self.assertEqual(science.pixel_has_data(y, x), dx + dy > 3)

    def test_psi_and_phi_image(self):
        p = PSF(0.00000001)  # A point function.
        img = LayeredImage("small_test", 6, 5, 2.0, 4.0, 10.0, p)

        # Create fake science and variance images.
        sci = img.get_science()
        var = img.get_variance()
        for y in range(6):
            for x in range(5):
                sci.set_pixel(y, x, float(x))
                var.set_pixel(y, x, float(y + 1))
        var.set_pixel(3, 1, KB_NO_DATA)

        # Generate and check psi and phi images.
        psi = img.generate_psi_image()
        self.assertEqual(psi.width, 5)
        self.assertEqual(psi.height, 6)

        phi = img.generate_phi_image()
        self.assertEqual(phi.width, 5)
        self.assertEqual(phi.height, 6)

        for y in range(6):
            for x in range(5):
                has_data = not (x == 1 and y == 3)
                self.assertEqual(psi.pixel_has_data(y, x), has_data)
                self.assertEqual(phi.pixel_has_data(y, x), has_data)
                if x != 1 or y != 3:
                    self.assertAlmostEqual(psi.get_pixel(y, x), x / (y + 1))
                    self.assertAlmostEqual(phi.get_pixel(y, x), 1.0 / (y + 1))

    def test_subtract_template(self):
        sci = self.image.get_science()
        sci.set_pixel(10, 7, KB_NO_DATA)
        sci.set_pixel(10, 21, KB_NO_DATA)
        old_sci = RawImage(sci.image.copy())  # Make a copy.

        template = RawImage(self.image.get_height(), self.image.get_width())
        template.set_all(0.0)
        for h in range(sci.height):
            # this doesn't raise if index is out of bounds....
            template.set_pixel(10, h, 0.01 * h)
        self.image.sub_template(template)

        for x in range(sci.width):
            for y in range(sci.height):
                val1 = old_sci.get_pixel(y, x)
                val2 = sci.get_pixel(y, x)
                if y == 10 and x != 7 and x != 21:
                    try:
                        self.assertAlmostEqual(val2, val1 - 0.01*x, delta=1e-6)
                    except AssertionError:
                        breakpoint()
                        a = 1
                else:
                    self.assertEqual(val1, val2)

    def test_read_write_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_layered_test_data"
            full_path = "%s/%s.fits" % (dir_name, file_name)
            im1 = LayeredImage(
                file_name,
                15,  # dim_x = 15 pixels,
                20,  # dim_y = 20 pixels,
                2.0,  # noise_level
                4.0,  # variance
                10.0,  # time = 10.0
                self.p,
            )

            # Make some changes to the mask to ensure that
            # layer has something to compare.
            mask1 = im1.get_mask()
            mask1.set_pixel(3, 5, 1.0)
            mask1.set_pixel(5, 3, 1.0)

            # Save the test data.
            im1.save_layers(dir_name + "/")

            # Reload the test data and check that it matches.
            im2 = LayeredImage(full_path, self.p)
            self.assertEqual(im1.get_height(), im2.get_height())
            self.assertEqual(im1.get_width(), im2.get_width())
            self.assertEqual(im1.get_npixels(), im2.get_npixels())
            self.assertEqual(im1.get_obstime(), im2.get_obstime())

            sci1 = im1.get_science()
            sci2 = im2.get_science()
            self.assertEqual(sci1.obstime, sci2.obstime)

            var1 = im1.get_variance()
            mask1 = im1.get_mask()
            var2 = im2.get_variance()
            mask2 = im2.get_mask()
            for x in range(im1.get_width()):
                for y in range(im2.get_height()):
                    self.assertEqual(sci1.get_pixel(y, x), sci2.get_pixel(y, x))
                    self.assertEqual(var1.get_pixel(y, x), var2.get_pixel(y, x))
                    self.assertEqual(mask1.get_pixel(y, x), mask2.get_pixel(y, x))

    def test_overwrite_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_layered_test_data2"
            full_path = "%s/%s.fits" % (dir_name, file_name)

            # Save the test image.
            img1 = LayeredImage(file_name, 15, 20, 2.0, 4.0, 10.0, self.p)
            img1.save_layers(dir_name + "/")
            with fits.open(full_path) as hdulist:
                self.assertEqual(len(hdulist), 4)
                self.assertEqual(hdulist[1].header["NAXIS1"], 20)
                self.assertEqual(hdulist[1].header["NAXIS2"], 15)

            # Save a new test image over the first and check
            # that it replaces it.
            img2 = LayeredImage(file_name, 25, 40, 2.0, 4.0, 10.0, self.p)
            img2.save_layers(dir_name + "/")
            with fits.open(full_path) as hdulist2:
                self.assertEqual(len(hdulist2), 4)
                self.assertEqual(hdulist2[1].header["NAXIS1"], 40)
                self.assertEqual(hdulist2[1].header["NAXIS2"], 25)


if __name__ == "__main__":
    unittest.main()
