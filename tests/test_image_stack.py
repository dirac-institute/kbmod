from kbmod import *
import tempfile
import unittest

class test_image_stack(unittest.TestCase):

    def setUp(self):
        # Create multiple fake layered images to use.
        self.num_images = 5
        self.images = [None] * self.num_images
        self.p = [None] * self.num_images
        for i in range(self.num_images):
            self.p[i] = psf(5.0 / float(2 * i + 1))
            self.images[i] = layered_image(("layered_test_%i" % i),
                                           80,    # dim_x = 80 pixels,
                                           60,    # dim_y = 60 pixels,
                                           2.0,   # noise_level
                                           4.0,   # variance
                                           2.0 * i,  # time
                                           self.p[i])

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
        self.assertEqual(len(self.im_stack.get_sciences()),
                         self.num_images)
        self.assertEqual(len(self.im_stack.get_variances()),
                         self.num_images)
        self.assertEqual(len(self.im_stack.get_masks()),
                         self.num_images)

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
        sci_stack = self.im_stack.get_sciences()
        for i in range(self.num_images):
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    self.assertTrue(sci_stack[i].pixel_has_data(x, y))

        self.im_stack.apply_mask_flags(1, [])

        # Check that one pixel is masked in each time.
        sci_stack = self.im_stack.get_sciences()
        for i in range(self.num_images):
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    if x == 10 and y == 10 + i:
                        self.assertFalse(sci_stack[i].pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci_stack[i].pixel_has_data(x, y))

    def test_create_master_mask(self):
        # Before we apply the master mask it defaults to all zero.
        # NOTE: This is current behavior, but might not be what we
        # actually want.
        master_mask = self.im_stack.get_master_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                self.assertEqual(master_mask.get_pixel(x, y), 0.0)

        # Apply the master mask for flag=1 and a threshold of the bit set
        # in at least one mask.
        self.im_stack.apply_master_mask(1, 1)

        # Check that the correct pixels are masked in each time.
        sci_stack = self.im_stack.get_sciences()
        for i in range(self.num_images):
            for y in range(self.im_stack.get_height()):
                for x in range(self.im_stack.get_width()):
                    if x == 10 and y >= 10 and y <= 10 + (self.num_images - 1):
                        self.assertFalse(sci_stack[i].pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci_stack[i].pixel_has_data(x, y))

        # Check that the master mask is now set.
        master_mask = self.im_stack.get_master_mask()
        for y in range(self.im_stack.get_height()):
            for x in range(self.im_stack.get_width()):
                if x == 10 and y >= 10 and y <= 10 + (self.num_images - 1):
                    self.assertEqual(master_mask.get_pixel(x, y), 1.0)
                else:
                    self.assertEqual(master_mask.get_pixel(x, y), 0.0)

    def test_subtract_template(self):
        width = 5
        height = 6
        p = psf(1.0)

        # Create three small images with known science pixels.
        images = []
        for i in range(3):
            image = layered_image(("layered_test_%i" % i), width, height,
                                  2.0, 4.0, 2.0 * i, p)
            sci_layer = image.get_science()
            for x in range(width):
                for y in range(height):
                    sci_layer.set_pixel(x, y, 10.0*i + 0.5*y)
            image.set_science(sci_layer)
            images.append(image)

        # Compute the simple difference.
        img_stack = image_stack(images)
        img_stack.simple_difference()

        # Check that the average for pixel (x, y) has been subtracted
        # out of each science image.
        sciences = img_stack.get_sciences()
        for i in range(3):
            for x in range(width):
                for y in range(height):
                    self.assertEqual(sciences[i].get_pixel(x, y), 10.0*(i-1))

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

if __name__ == '__main__':
   unittest.main()

