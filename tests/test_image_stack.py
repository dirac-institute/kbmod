import unittest

from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import make_fake_layered_image
from kbmod.search import *


class test_ImageStack(unittest.TestCase):
    def setUp(self):
        # Create multiple fake layered images to use.
        self.num_images = 5
        self.images = [None] * self.num_images
        self.p = [None] * self.num_images
        for i in range(self.num_images):
            self.p[i] = PSF.make_gaussian_kernel(5.0 / float(2 * i + 1))
            self.images[i] = make_fake_layered_image(
                60,  # dim_x = 60 pixels,
                80,  # dim_y = 80 pixels,
                2.0,  # noise_level
                4.0,  # variance
                2.0 * i + 1.0,  # time
                self.p[i],
            )

            # Include one masked pixel per time step at (10, 10 + i).
            mask = self.images[i].mask
            mask[10, 10 + i] = 1

        self.im_stack = ImageStack(self.images)

    def test_create_empty(self):
        im_stack = ImageStack([])
        self.assertEqual(len(im_stack), 0)
        self.assertEqual(im_stack.width, 0)
        self.assertEqual(im_stack.height, 0)

        im_stack2 = ImageStack()
        self.assertEqual(len(im_stack2), 0)
        self.assertEqual(im_stack2.width, 0)
        self.assertEqual(im_stack2.height, 0)

    def test_create(self):
        self.assertEqual(len(self.im_stack), self.num_images)
        self.assertEqual(self.num_images, self.im_stack.num_times)
        self.assertEqual(self.im_stack.height, 80)
        self.assertEqual(self.im_stack.width, 60)
        self.assertEqual(self.im_stack.get_npixels(), 60 * 80)
        self.assertEqual(self.im_stack.get_total_pixels(), 5 * 60 * 80)

        # Add an image of the wrong size.
        self.images.append(make_fake_layered_image(5, 5, 2.0, 4.0, 10.0, self.p[0]))
        with self.assertRaises(RuntimeError):
            _ = ImageStack(self.images)

    def test_access(self):
        """Test we can access an individual image."""
        img = self.im_stack.get_single_image(1)
        self.assertEqual(img.time, 3.0)

        # Test an out of bounds access.
        with self.assertRaises(IndexError):
            img = self.im_stack.get_single_image(self.num_images + 1)

    def test_set(self):
        new_img = make_fake_layered_image(60, 80, 2.0, 4.0, -1.0, self.p[0])
        self.im_stack.set_single_image(2, new_img)
        self.assertEqual(self.im_stack.get_obstime(2), -1.0)

        # Test an out of bounds access.
        with self.assertRaises(IndexError):
            img = self.im_stack.set_single_image(self.num_images + 1, new_img)

        # Test an invalid shaped image.
        new_img2 = make_fake_layered_image(10, 80, 2.0, 4.0, -1.0, self.p[0])
        with self.assertRaises(RuntimeError):
            self.im_stack.set_single_image(2, new_img2)

    def test_append(self):
        new_img = make_fake_layered_image(60, 80, 2.0, 4.0, -1.0, self.p[0])
        self.im_stack.append_image(new_img)
        self.assertEqual(len(self.im_stack), self.num_images + 1)
        self.assertEqual(self.im_stack.get_obstime(self.num_images), -1.0)

        # Test an invalid shaped image.
        new_img2 = make_fake_layered_image(10, 80, 2.0, 4.0, -1.0, self.p[0])
        with self.assertRaises(RuntimeError):
            self.im_stack.append_image(new_img2)

    def test_times(self):
        """Check that we can access specific times.
        Check that we can build the full zeroed times list."""
        self.assertEqual(self.im_stack.get_obstime(1), 3.0)

        # Check that we can build the full zeroed times list.
        times = self.im_stack.build_zeroed_times()
        self.assertEqual(len(times), self.num_images)
        for i in range(self.num_images):
            self.assertEqual(times[i], 2.0 * i)

    def test_sort_by_time(self):
        local_psf = PSF.make_gaussian_kernel(1.0)
        images = [
            make_fake_layered_image(10, 15, 0.0, 0.0, 3.0, local_psf),
            make_fake_layered_image(10, 15, 0.0, 0.0, 1.0, local_psf),
            make_fake_layered_image(10, 15, 0.0, 0.0, 4.0, local_psf),
            make_fake_layered_image(10, 15, 0.0, 0.0, 5.0, local_psf),
            make_fake_layered_image(10, 15, 0.0, 0.0, 2.0, local_psf),
        ]
        im_stack = ImageStack(images)
        self.assertListEqual(im_stack.zeroed_times, [0.0, -2.0, 1.0, 2.0, -1.0])

        im_stack.sort_by_time()
        self.assertListEqual(im_stack.zeroed_times, [0.0, 1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
