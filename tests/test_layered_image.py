import tempfile
import unittest

from astropy.io import fits

from kbmod.search import *


class test_layered_image(unittest.TestCase):
    def setUp(self):
        self.p = psf(1.0)

        # Create a fake layered image to use.
        self.image = layered_image(
            "layered_test",
            80,  # dim_x = 80 pixels,
            60,  # dim_y = 60 pixels,
            2.0,  # noise_level
            4.0,  # variance
            10.0,  # time = 10.0
            self.p,
        )

    def test_create(self):
        self.assertIsNotNone(self.image)
        self.assertEqual(self.image.get_width(), 80)
        self.assertEqual(self.image.get_height(), 60)
        self.assertEqual(self.image.get_npixels(), 80 * 60)
        self.assertEqual(self.image.get_obstime(), 10.0)
        self.assertEqual(self.image.get_name(), "layered_test")

        # Create a fake layered_image.
        science = self.image.get_science()
        variance = self.image.get_variance()
        mask = self.image.get_mask()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                self.assertEqual(mask.get_pixel(x, y), 0)
                self.assertEqual(variance.get_pixel(x, y), 4.0)

                # These will be potentially flakey due to the random
                # creation (but with very low probability).
                self.assertGreaterEqual(science.get_pixel(x, y), -100.0)
                self.assertLessEqual(science.get_pixel(x, y), 100.0)

    def test_create_from_layers(self):
        sci = raw_image(30, 40)
        for y in range(40):
            for x in range(30):
                sci.set_pixel(x, y, x + 30.0 * y)

        var = raw_image(30, 40)
        var.set_all(1.0)

        mask = raw_image(30, 40)
        mask.set_all(0.0)

        # Create the layered image.
        img2 = layered_image(sci, var, mask, psf(2.0))
        self.assertEqual(img2.get_width(), 30)
        self.assertEqual(img2.get_height(), 40)
        self.assertEqual(img2.get_npixels(), 30.0 * 40.0)
        self.assertEqual(img2.get_obstime(), -1.0)  # No time given

        # Check the layers.
        science = img2.get_science()
        variance = img2.get_variance()
        mask2 = img2.get_mask()
        for y in range(img2.get_height()):
            for x in range(img2.get_width()):
                self.assertEqual(mask2.get_pixel(x, y), 0)
                self.assertEqual(variance.get_pixel(x, y), 1.0)
                self.assertAlmostEqual(science.get_pixel(x, y), x + 30.0 * y)

    def test_add_object(self):
        science = self.image.get_science()
        science_50_50 = science.get_pixel(50, 50)
        self.image.add_object(50, 50, 500.0)

        science = self.image.get_science()
        self.assertLess(science_50_50, science.get_pixel(50, 50))

    def test_overwrite_psf(self):
        p1 = self.image.get_psf()
        self.assertEqual(p1.get_size(), 25)
        self.assertEqual(p1.get_dim(), 5)
        self.assertEqual(p1.get_radius(), 2)

        psq1 = self.image.get_psfsq()
        self.assertEqual(psq1.get_size(), 25)
        self.assertEqual(psq1.get_dim(), 5)
        self.assertEqual(psq1.get_radius(), 2)

        # Get the science pixel with the original PSF blurring.
        science_org = self.image.get_science()
        self.image.add_object(50, 50, 500.0)
        science_pixel_psf1 = self.image.get_science().get_pixel(50, 50)

        # Change the PSF.
        self.image.set_psf(psf(0.0001))

        # Check that we retrieve the correct PSF.
        p2 = self.image.get_psf()
        self.assertEqual(p2.get_size(), 1)
        self.assertEqual(p2.get_dim(), 1)
        self.assertEqual(p2.get_radius(), 0)

        psq2 = self.image.get_psfsq()
        self.assertEqual(psq2.get_size(), 1)
        self.assertEqual(psq2.get_dim(), 1)
        self.assertEqual(psq2.get_radius(), 0)

        # Check that the science pixel with the new PSF blurring is
        # larger (because the PSF is tighter).
        self.image.set_science(science_org)
        self.image.add_object(50, 50, 500.0)
        science_pixel_psf2 = self.image.get_science().get_pixel(50, 50)
        self.assertLess(science_pixel_psf1, science_pixel_psf2)

    def test_mask_threshold(self):
        masked_pixels = {}
        threshold = 20.0

        # Add an object brighter than the threshold.
        self.image.add_object(50, 50, 500.0)

        # Find all the pixels that should be masked.
        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                value = science.get_pixel(x, y)
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
                index = self.image.get_width() * y + x
                if index in masked_pixels:
                    self.assertFalse(science.pixel_has_data(x, y))
                else:
                    self.assertTrue(science.pixel_has_data(x, y))

    def test_apply_mask(self):
        # Nothing is initially masked.
        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                self.assertTrue(science.pixel_has_data(x, y))

        # Mask out three pixels.
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 2)
        mask.set_pixel(10, 13, 3)
        self.image.set_mask(mask)

        # Apply the mask flags to only (10, 11) and (10, 13)
        self.image.apply_mask_flags(1, [])

        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                if x == 10 and (y == 11 or y == 13):
                    self.assertFalse(science.pixel_has_data(x, y))
                else:
                    self.assertTrue(science.pixel_has_data(x, y))

    def test_apply_mask_exceptions(self):
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 2)
        mask.set_pixel(10, 13, 3)
        self.image.set_mask(mask)

        # Apply the mask flags to only (10, 11).
        self.image.apply_mask_flags(1, [1])

        science = self.image.get_science()
        for y in range(self.image.get_height()):
            for x in range(self.image.get_width()):
                if x == 10 and y == 13:
                    self.assertFalse(science.pixel_has_data(x, y))
                else:
                    self.assertTrue(science.pixel_has_data(x, y))

    def test_grow_mask(self):
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 1)
        mask.set_pixel(10, 13, 1)
        self.image.set_mask(mask)
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
                self.assertEqual(science.pixel_has_data(x, y), not should_mask)

    def test_grow_mask_mult(self):
        mask = self.image.get_mask()
        mask.set_pixel(10, 11, 1)
        mask.set_pixel(10, 12, 1)
        self.image.set_mask(mask)
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
                self.assertEqual(science.pixel_has_data(x, y), dx + dy > 3)

    def test_psi_and_phi_image(self):
        p = psf(0.00000001)  # A point function.
        img = layered_image("small_test", 6, 5, 2.0, 4.0, 10.0, p)

        # Create fake science and variance images.
        sci = img.get_science()
        var = img.get_variance()
        for x in range(6):
            for y in range(5):
                sci.set_pixel(x, y, float(x))
                var.set_pixel(x, y, float(y + 1))
        var.set_pixel(3, 1, KB_NO_DATA)
        img.set_science(sci)
        img.set_variance(var)

        # Generate and check psi and phi images.
        psi = img.generate_psi_image()
        self.assertEqual(psi.get_width(), 6)
        self.assertEqual(psi.get_height(), 5)

        phi = img.generate_phi_image()
        self.assertEqual(phi.get_width(), 6)
        self.assertEqual(phi.get_height(), 5)

        for x in range(6):
            for y in range(5):
                has_data = not (x == 3 and y == 1)
                self.assertEqual(psi.pixel_has_data(x, y), has_data)
                self.assertEqual(phi.pixel_has_data(x, y), has_data)
                if x != 3 or y != 1:
                    self.assertAlmostEqual(psi.get_pixel(x, y), float(x) / float(y + 1))
                    self.assertAlmostEqual(phi.get_pixel(x, y), 1.0 / float(y + 1))

    def test_subtract_template(self):
        old_science = self.image.get_science()

        # Mask out a few points and reset (needed because of how pybind handles
        # pass by reference).
        old_science.set_pixel(5, 6, KB_NO_DATA)
        old_science.set_pixel(10, 7, KB_NO_DATA)
        old_science.set_pixel(10, 21, KB_NO_DATA)
        self.image.set_science(old_science)

        template = raw_image(self.image.get_width(), self.image.get_height())
        template.set_all(0.0)
        for h in range(old_science.get_height()):
            template.set_pixel(10, h, 0.01 * h)
        self.image.sub_template(template)

        new_science = self.image.get_science()
        for x in range(old_science.get_width()):
            for y in range(old_science.get_height()):
                val1 = old_science.get_pixel(x, y)
                val2 = new_science.get_pixel(x, y)
                if x == 10 and y != 7 and y != 21:
                    self.assertAlmostEqual(val2, val1 - 0.01 * y, delta=1e-6)
                else:
                    self.assertEqual(val1, val2)

    def test_read_write_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_layered_test_data"
            full_path = "%s/%s.fits" % (dir_name, file_name)
            im1 = layered_image(
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
            im1.set_mask(mask1)

            # Save the test data.
            im1.save_layers(dir_name + "/")

            # Reload the test data and check that it matches.
            im2 = layered_image(full_path, self.p)
            self.assertEqual(im1.get_height(), im2.get_height())
            self.assertEqual(im1.get_width(), im2.get_width())
            self.assertEqual(im1.get_npixels(), im2.get_npixels())
            self.assertEqual(im1.get_obstime(), im2.get_obstime())

            sci1 = im1.get_science()
            sci2 = im2.get_science()
            self.assertEqual(sci1.get_obstime(), sci2.get_obstime())
            
            var1 = im1.get_variance()
            mask1 = im1.get_mask()
            var2 = im2.get_variance()
            mask2 = im2.get_mask()
            for x in range(im1.get_width()):
                for y in range(im2.get_height()):
                    self.assertEqual(sci1.get_pixel(x, y), sci2.get_pixel(x, y))
                    self.assertEqual(var1.get_pixel(x, y), var2.get_pixel(x, y))
                    self.assertEqual(mask1.get_pixel(x, y), mask2.get_pixel(x, y))


    def test_overwrite_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_layered_test_data2"
            full_path = "%s/%s.fits" % (dir_name, file_name)

            # Save the test image.
            img1 = layered_image(file_name, 15, 20, 2.0, 4.0, 10.0, self.p)
            img1.save_layers(dir_name + "/")
            with fits.open(full_path) as hdulist:
                self.assertEqual(len(hdulist), 4)
                self.assertEqual(hdulist[1].header["NAXIS1"], 15)
                self.assertEqual(hdulist[1].header["NAXIS2"], 20)

            # Save a new test image over the first and check
            # that it replaces it.
            img2 = layered_image(file_name, 25, 40, 2.0, 4.0, 10.0, self.p)
            img2.save_layers(dir_name + "/")
            with fits.open(full_path) as hdulist2:
                self.assertEqual(len(hdulist2), 4)
                self.assertEqual(hdulist2[1].header["NAXIS1"], 25)
                self.assertEqual(hdulist2[1].header["NAXIS2"], 40)


if __name__ == "__main__":
    unittest.main()
