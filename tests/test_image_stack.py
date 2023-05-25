import tempfile
import unittest

from kbmod.search import *


class test_image_stack(unittest.TestCase):
    def setUp(self):
        # Create multiple fake layered images to use.
        self.num_images = 5
        self.images = [None] * self.num_images
        self.p = [None] * self.num_images
        for i in range(self.num_images):
            self.p[i] = psf(5.0 / float(2 * i + 1))
            self.images[i] = layered_image(
                ("layered_test_%i" % i),
                80,  # dim_x = 80 pixels,
                60,  # dim_y = 60 pixels,
                2.0,  # noise_level
                4.0,  # variance
                2.0 * i,  # time
                self.p[i],
            )

            # Include one masked pixel per time step at (10, 10 + i).
            mask = self.images[i].get_mask()
            mask.set_pixel(10, 10 + i, 1)
            self.images[i].set_mask(mask)

        self.im_stack = image_stack(self.images)

    def test_create(self):
        self.assertEqual(self.num_images, self.im_stack.img_count())
        self.assertEqual(self.im_stack.get_height(), 60)
        self.assertEqual(self.im_stack.get_width(), 80)
        self.assertEqual(self.im_stack.get_ppi(), 60 * 80)

    def test_times(self):
        times = self.im_stack.get_times()
        self.assertEqual(len(times), self.num_images)
        for i in range(self.num_images):
            self.assertEqual(times[i], 2.0 * i)

        new_times = [3.0 * i for i in range(self.num_images)]
        self.im_stack.set_times(new_times)

        times2 = self.im_stack.get_times()
        self.assertEqual(len(times2), self.num_images)
        for i in range(self.num_images):
            self.assertEqual(times2[i], 3.0 * i)

    def test_apply_mask(self):
        # Nothing is initially masked.
        for i in range(self.num_images):
            sci = self.im_stack.get_single_image(i).get_science()
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    self.assertTrue(sci.pixel_has_data(x, y))

        self.im_stack.apply_mask_flags(1, [])

        # Check that one pixel is masked in each time.
        for i in range(self.num_images):
            sci = self.im_stack.get_single_image(i).get_science()
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    if x == 10 and y == 10 + i:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_create_global_mask(self):
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(global_mask.get_pixel(x, y), 0.0)

        # Apply the global mask for flag=1 and a threshold of the bit set
        # in at least one mask.
        self.im_stack.apply_global_mask(1, 1)

        # Check that the correct pixels are masked in each time.
        for i in range(self.num_images):
            sci = self.im_stack.get_single_image(i).get_science()
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    if x == 10 and y >= 10 and y <= 10 + (self.num_images - 1):
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

        # Check that the global mask is now set.
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                if x == 10 and y >= 10 and y <= 10 + (self.num_images - 1):
                    self.assertEqual(global_mask.get_pixel(x, y), 1.0)
                else:
                    self.assertEqual(global_mask.get_pixel(x, y), 0.0)

    def test_create_global_mask_reset(self):
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(global_mask.get_pixel(x, y), 0.0)

        # Apply the global mask for flag=1 and a threshold of the bit set
        # in at least one mask.
        self.im_stack.apply_global_mask(1, 1)

        # Check that the global mask is set.
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                if x == 10 and y >= 10 and y <= 10 + (self.num_images - 1):
                    self.assertEqual(global_mask.get_pixel(x, y), 1.0)
                else:
                    self.assertEqual(global_mask.get_pixel(x, y), 0.0)

        # Unmask the pixels.
        for i in range(self.num_images):
            img = self.im_stack.get_single_image(i)
            mask = img.get_mask()
            mask.set_pixel(10, 10 + i, 0)
            img.set_mask(mask)
            self.im_stack.set_single_image(i, img)

        # Reapply the mask and check that nothing is masked.
        # Note the science pixels will still be masked from the previous application.
        self.im_stack.apply_global_mask(1, 1)
        global_mask = self.im_stack.get_global_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(global_mask.get_pixel(x, y), 0.0)

    def test_subtract_template(self):
        width = 5
        height = 6
        p = psf(1.0)

        # Create three small images with known science pixels.
        images = []
        for i in range(3):
            image = layered_image(("layered_test_%i" % i), width, height, 2.0, 4.0, 2.0 * i, p)
            sci_layer = image.get_science()
            for x in range(width):
                for y in range(height):
                    if x == 4 and y <= i:
                        sci_layer.set_pixel(x, y, KB_NO_DATA)
                    else:
                        sci_layer.set_pixel(x, y, 10.0 * i + 0.5 * y)
            image.set_science(sci_layer)
            images.append(image)

        # Compute the simple difference.
        img_stack = image_stack(images)
        img_stack.simple_difference()

        # Check that the average for pixel (x, y) has been subtracted
        # from each science image. Start with the rows of unmasked pixels.
        for i in range(3):
            sci = img_stack.get_single_image(i).get_science()
            for x in range(width - 1):
                for y in range(height):
                    self.assertEqual(sci.get_pixel(x, y), 10.0 * (i - 1))

        # Check the masked out pixels.
        sci0 = img_stack.get_single_image(0).get_science()
        self.assertEqual(sci0.get_pixel(4, 0), KB_NO_DATA)
        self.assertEqual(sci0.get_pixel(4, 1), 0.0)
        self.assertEqual(sci0.get_pixel(4, 2), -5.0)
        self.assertEqual(sci0.get_pixel(4, 3), -10.0)

        sci1 = img_stack.get_single_image(1).get_science()
        self.assertEqual(sci1.get_pixel(4, 0), KB_NO_DATA)
        self.assertEqual(sci1.get_pixel(4, 1), KB_NO_DATA)
        self.assertEqual(sci1.get_pixel(4, 2), 5.0)
        self.assertEqual(sci1.get_pixel(4, 3), 0.0)

        sci2 = img_stack.get_single_image(2).get_science()
        self.assertEqual(sci2.get_pixel(4, 0), KB_NO_DATA)
        self.assertEqual(sci2.get_pixel(4, 1), KB_NO_DATA)
        self.assertEqual(sci2.get_pixel(4, 2), KB_NO_DATA)
        self.assertEqual(sci2.get_pixel(4, 3), 10.0)

    def test_different_psfs(self):
        # Add a stationary fake object to each image. Then test that
        # the flux at each time is monotonically increasing (because
        # the PSF is getting tighter).
        last_val = -100.0
        for i in range(self.num_images):
            img = self.im_stack.get_single_image(i)
            img.add_object(10, 20, 500.0)

            sci = img.get_science()
            pix_val = sci.get_pixel(10, 20)
            self.assertGreater(pix_val, last_val)
            last_val = pix_val

    def test_simple_shift_and_stack(self):
        width = 30
        height = 40
        p = psf(1)

        # Create multiple fake layered images to use.
        num_images = 3
        images = [None] * num_images
        for i in range(num_images):
            images[i] = layered_image(
                ("layered_test_%i" % i),
                width,  # dim_x = 30 pixels,
                height,  # dim_y = 40 pixels,
                0.0,  # noise_level
                0.0,  # variance
                float(i),  # time
                p,
            )

            # Set one science pixel to 50 in each image.
            sci = images[i].get_science()
            sci.set_pixel(10, 10 + 2 * i, 50.0)
            images[i].set_science(sci)
        im_stack = image_stack(images)

        # With no motion, the three bright pixels should by at
        # different places.
        img00 = im_stack.simple_shift_and_stack(0.0, 0.0, False)
        self.assertEqual(img00.get_width(), width)
        self.assertEqual(img00.get_height(), height)
        for x in range(width):
            for y in range(height):
                if x == 10 and (y in [10, 12, 14]):
                    self.assertAlmostEqual(img00.get_pixel(x, y), 50.0)
                else:
                    self.assertAlmostEqual(img00.get_pixel(x, y), 0.0)

        # With motion y_v = 2, the three bright pixels should line up.
        img02 = im_stack.simple_shift_and_stack(0.0, 2.0, False)
        self.assertEqual(img02.get_width(), width)
        self.assertEqual(img02.get_height(), height)
        for x in range(width):
            for y in range(height):
                if x == 10 and y == 10:
                    self.assertAlmostEqual(img02.get_pixel(x, y), 150.0)
                else:
                    self.assertAlmostEqual(img02.get_pixel(x, y), 0.0)

        # With motion y_v = 2, the three bright pixels should line up.
        # Test the mean.
        img02 = im_stack.simple_shift_and_stack(0.0, 2.0, True)
        self.assertEqual(img02.get_width(), width)
        self.assertEqual(img02.get_height(), height)
        for x in range(width):
            for y in range(height):
                if x == 10 and y == 10:
                    self.assertAlmostEqual(img02.get_pixel(x, y), 50.0)
                else:
                    self.assertAlmostEqual(img02.get_pixel(x, y), 0.0)

        # With motion x_v = 1, y_v = 2, the three bright pixels should
        # again be in different places.
        img12 = im_stack.simple_shift_and_stack(1.0, 2.0, False)
        self.assertEqual(img12.get_width(), width)
        self.assertEqual(img12.get_height(), height)
        for x in range(width):
            for y in range(height):
                if y == 10 and x in [8, 9, 10]:
                    self.assertAlmostEqual(img12.get_pixel(x, y), 50.0)
                else:
                    self.assertAlmostEqual(img12.get_pixel(x, y), 0.0)


if __name__ == "__main__":
    unittest.main()
