import tempfile
import unittest

from kbmod.fake_data_creator import add_fake_object
from kbmod.search import *


class test_ImageStack(unittest.TestCase):
    def setUp(self):
        # Create multiple fake layered images to use.
        self.num_images = 5
        self.images = [None] * self.num_images
        self.p = [None] * self.num_images
        for i in range(self.num_images):
            self.p[i] = PSF(5.0 / float(2 * i + 1))
            self.images[i] = LayeredImage(
                ("layered_test_%i" % i),
                60,  # dim_x = 60 pixels,
                80,  # dim_y = 80 pixels,
                2.0,  # noise_level
                4.0,  # variance
                2.0 * i + 1.0,  # time
                self.p[i],
            )

            # Include one masked pixel per time step at (10, 10 + i).
            mask = self.images[i].get_mask()
            mask.set_pixel(10, 10 + i, 1)

        self.im_stack = ImageStack(self.images)

    def test_create(self):
        self.assertEqual(self.num_images, self.im_stack.img_count())
        self.assertEqual(self.im_stack.get_height(), 80)
        self.assertEqual(self.im_stack.get_width(), 60)
        self.assertEqual(self.im_stack.get_npixels(), 60 * 80)

    def test_access(self):
        """Test we can access an individual image."""
        img = self.im_stack.get_single_image(1)
        self.assertEqual(img.get_obstime(), 3.0)
        self.assertEqual(img.get_name(), "layered_test_1")

        # Test an out of bounds access.
        with self.assertRaises(IndexError):
            img = self.im_stack.get_single_image(self.num_images + 1)

    def test_times(self):
        """Check that we can access specific times.
        Check that we can build the full zeroed times list."""
        self.assertEqual(self.im_stack.get_obstime(1), 3.0)
        self.assertEqual(self.im_stack.get_zeroed_time(1), 2.0)

        # Check that we can build the full zeroed times list.
        times = self.im_stack.build_zeroed_times()
        self.assertEqual(len(times), self.num_images)
        for i in range(self.num_images):
            self.assertEqual(times[i], 2.0 * i)

    def test_apply_mask(self):
        # Nothing is initially masked.
        for i in range(self.num_images):
            sci = self.im_stack.get_single_image(i).get_science()
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    self.assertTrue(sci.pixel_has_data(y, x))

        self.im_stack.apply_mask_flags(1, [])

        # Check that one pixel is masked in each time.
        for i in range(self.num_images):
            sci = self.im_stack.get_single_image(i).get_science()
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    if y == 10 and x == 10 + i:
                        self.assertFalse(sci.pixel_has_data(y, x))
                    else:
                        self.assertTrue(sci.pixel_has_data(y, x))

    def test_create_global_mask(self):
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(global_mask.get_pixel(y, x), 0.0)

        # Apply the global mask for flag=1 and a threshold of the bit set
        # in at least one mask.
        self.im_stack.apply_global_mask(1, 1)

        # Check that the correct pixels are masked in each time.
        for i in range(self.num_images):
            sci = self.im_stack.get_single_image(i).get_science()
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    if y == 10 and x >= 10 and x <= 10 + (self.num_images - 1):
                        self.assertFalse(sci.pixel_has_data(y, x))
                    else:
                        self.assertTrue(sci.pixel_has_data(y, x))

        # Check that the global mask is now set.
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                if y == 10 and x >= 10 and x <= 10 + (self.num_images - 1):
                    self.assertEqual(global_mask.get_pixel(y, x), 1.0)
                else:
                    self.assertEqual(global_mask.get_pixel(y, x), 0.0)

    def test_create_global_mask_reset(self):
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(global_mask.get_pixel(y, x), 0.0)

        # Apply the global mask for flag=1 and a threshold of the bit set
        # in at least one mask.
        self.im_stack.apply_global_mask(1, 1)

        # Check that the global mask is set.
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                if y == 10 and x >= 10 and x <= 10 + (self.num_images - 1):
                    self.assertEqual(global_mask.get_pixel(y, x), 1.0)
                else:
                    self.assertEqual(global_mask.get_pixel(y, x), 0.0)

        # Unmask the pixels.
        for i in range(self.num_images):
            img = self.im_stack.get_single_image(i)
            mask = img.get_mask()
            mask.set_pixel(10, 10 + i, 0)

        # Reapply the mask and check that nothing is masked.
        # Note the science pixels will still be masked from the previous application.
        self.im_stack.apply_global_mask(1, 1)
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(global_mask.get_pixel(y, x), 0.0)

    # WOW, this is the first test that caught the fact that interpolated_add
    # called add, and that add had flipped i and j by accident. The first one.
    # TODO: more clean understandable tests for basic functionality, these big
    # are super hard to debug....
    def test_different_psfs(self):
        # Add a stationary fake object to each image. Then test that
        # the flux at each time is monotonically increasing (because
        # the PSF is getting tighter).
        last_val = -100.0
        for i in range(self.num_images):
            img = self.im_stack.get_single_image(i)
            add_fake_object(img, 10, 20, 500.0, self.p[i])
            sci = img.get_science()
            pix_val = sci.get_pixel(20, 10)
            self.assertGreater(pix_val, last_val)
            last_val = pix_val


if __name__ == "__main__":
    unittest.main()
