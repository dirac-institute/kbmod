import logging
import numpy as np
import unittest

from kbmod.core.image_stack_py import (
    image_stack_add_fake_object,
    make_fake_image_stack,
    ImageStackPy,
    LayeredImagePy,
)
from kbmod.core.psf import PSF


class test_layered_image_py(unittest.TestCase):
    def test_create_layered_image_py(self):
        """Test that we can create a LayeredImagePy"""
        height = 20
        width = 15

        sci = np.full((height, width), 1.0)
        var = np.full((height, width), 0.1)
        msk = np.zeros((height, width))
        img = LayeredImagePy(sci, var, msk)

        self.assertEqual(img.width, width)
        self.assertEqual(img.height, height)
        self.assertTrue(np.allclose(img.sci, sci))
        self.assertTrue(np.allclose(img.var, var))
        self.assertTrue(np.allclose(img.mask, msk))
        self.assertTrue(np.allclose(img.psf, np.array([[1.0]])))


class test_image_stack_py(unittest.TestCase):
    def test_create_image_stack_py(self):
        """Test that we can create an ImageStackPy"""
        num_times = 10
        height = 20
        width = 15

        times = np.arange(num_times)
        sci = [np.full((height, width), 1.0) for _ in range(num_times)]
        var = [np.full((height, width), 0.1) for _ in range(num_times)]

        stack = ImageStackPy(times, sci, var)
        self.assertEqual(stack.num_times, num_times)
        self.assertEqual(stack.width, width)
        self.assertEqual(stack.height, height)
        self.assertEqual(stack.npixels, height * width)
        self.assertEqual(stack.total_pixels, num_times * height * width)
        self.assertEqual(stack.num_masked_pixels(), 0)

        self.assertTrue(np.allclose(stack.times, times))
        self.assertTrue(np.allclose(stack.zeroed_times, np.arange(num_times)))

        for idx in range(num_times):
            self.assertTrue(np.all(stack.sci[idx] == 1.0))
            self.assertTrue(np.all(stack.var[idx] == 0.1))
        self.assertEqual(len(stack.psfs), num_times)

        # We can append an image and a LayeredImage.
        stack.append_image(0, sci[0], var[0])
        self.assertEqual(stack.num_times, num_times + 1)
        self.assertEqual(len(stack.sci), num_times + 1)
        self.assertEqual(len(stack.var), num_times + 1)
        self.assertEqual(len(stack.psfs), num_times + 1)
        self.assertTrue(np.allclose(stack.times, np.append(times, 0.0)))

        layered = LayeredImagePy(sci[1], var[1], time=5.0)
        stack.append_layered_image(layered)
        self.assertEqual(stack.num_times, num_times + 2)
        self.assertEqual(len(stack.sci), num_times + 2)
        self.assertEqual(len(stack.var), num_times + 2)
        self.assertEqual(len(stack.psfs), num_times + 2)
        self.assertEqual(stack.times[num_times + 1], 5.0)

        # Test that we fail with bad input.
        self.assertRaises(ValueError, ImageStackPy, times)
        self.assertRaises(ValueError, ImageStackPy, times, [np.array([1.0])], var)
        self.assertRaises(ValueError, ImageStackPy, times, sci, [np.array([1.0])])

    def test_create_empty_image_stack_py(self):
        """Test that we can create an empty ImageStackPy"""
        stack = ImageStackPy()
        self.assertEqual(stack.num_times, 0)
        self.assertEqual(stack.width, -1)
        self.assertEqual(stack.height, -1)

        # We can append the first few images.
        num_times = 5
        height = 20
        width = 15
        for idx in range(num_times):
            sci = np.full((height, width), float(idx))
            var = np.full((height, width), 0.1 * float(idx))
            stack.append_image(float(idx + 5.0), sci, var)

            self.assertEqual(stack.num_times, idx + 1)
            self.assertEqual(stack.width, width)
            self.assertEqual(stack.height, height)
            self.assertTrue(np.allclose(stack.sci[idx], sci))
            self.assertTrue(np.allclose(stack.var[idx], var))
        self.assertTrue(np.allclose(stack.times, [5.0, 6.0, 7.0, 8.0, 9.0]))
        self.assertTrue(np.allclose(stack.zeroed_times, [0.0, 1.0, 2.0, 3.0, 4.0]))

        # We fail if we try to create an ImageStackPy with no times but other data.
        times = np.arange(num_times)
        sci = [np.full((height, width), float(i)) for i in range(num_times)]
        var = [np.full((height, width), 0.1 * i) for i in range(num_times)]
        self.assertRaises(ValueError, ImageStackPy, None, sci, var)

    def test_create_image_stack_from_nparray_py(self):
        """Test that we can create an ImageStackPy from a single 3-d numpy array."""
        num_times = 10
        height = 20
        width = 15

        times = np.arange(num_times)
        sci = np.full((num_times, height, width), 1.0)
        var = np.full((num_times, height, width), 0.1)

        stack = ImageStackPy(times, sci, var)
        self.assertEqual(stack.num_times, num_times)
        self.assertEqual(stack.width, width)
        self.assertEqual(stack.height, height)
        self.assertEqual(stack.npixels, height * width)
        self.assertEqual(stack.total_pixels, num_times * height * width)
        self.assertEqual(stack.num_masked_pixels(), 0)

        self.assertTrue(np.allclose(stack.times, times))
        self.assertTrue(np.allclose(stack.zeroed_times, np.arange(num_times)))

        for idx in range(num_times):
            self.assertTrue(np.all(stack.sci[idx] == 1.0))
            self.assertTrue(np.all(stack.var[idx] == 0.1))
        self.assertEqual(len(stack.psfs), num_times)

    def test_create_image_stack_py_masked(self):
        """Test that we can create an ImageStackPy with masked pixels."""
        num_times = 3
        height = 8
        width = 5

        times = np.arange(num_times) + 5
        sci = np.full((num_times, height, width), 1.0)
        var = np.full((num_times, height, width), 0.1)
        mask = np.zeros_like(sci)
        mask[0, 1, 2] = 1
        mask[0, 1, 3] = 1
        mask[2, 4, 4] = 1

        stack = ImageStackPy(times, sci, var, mask=mask)
        self.assertEqual(stack.num_times, num_times)
        self.assertEqual(stack.width, width)
        self.assertEqual(stack.height, height)
        self.assertEqual(stack.npixels, height * width)
        self.assertEqual(stack.total_pixels, num_times * height * width)
        self.assertEqual(stack.num_masked_pixels(), 3)

        self.assertTrue(np.allclose(stack.times, times))
        self.assertTrue(np.allclose(stack.zeroed_times, np.arange(num_times)))

        for idx in range(num_times):
            mask_mask = mask[idx] > 0

            self.assertTrue(np.all(stack.sci[idx][~mask_mask] == 1.0))
            self.assertTrue(np.all(np.isnan(stack.sci[idx][mask_mask])))

            self.assertTrue(np.all(stack.var[idx][~mask_mask] == 0.1))
            self.assertTrue(np.all(np.isnan(stack.var[idx][mask_mask])))

    def test_get_set_image_stack_py(self):
        """Test that we can get and set the data at a single time step of ImageStackPy"""
        num_times = 10
        height = 20
        width = 15

        times = np.arange(num_times)
        sci = [np.full((height, width), 1.0, dtype=np.float32) for _ in range(num_times)]
        var = [np.full((height, width), 0.1, dtype=np.float32) for _ in range(num_times)]
        stack = ImageStackPy(times, sci, var)

        img = stack.get_single_image(2)
        self.assertTrue(np.all(img.sci == 1.0))
        self.assertTrue(np.all(img.var == 0.1))
        self.assertEqual(img.width, width)
        self.assertEqual(img.height, height)
        self.assertEqual(img.time, 2.0)

        # Test that we can set the data at a single time step 5.
        sci5 = np.full((height, width), 2.0, dtype=np.float32)
        var5 = np.full((height, width), 0.2, dtype=np.float32)
        msk5 = np.zeros((height, width), dtype=np.float32)
        msk5[1, 2] = 1.0
        img5 = LayeredImagePy(sci5, var5, msk5, time=10.0)
        stack.set_single_image(5, img5)

        # Check that we have the expected data at time 5 (including
        # the correct pixels being masked).
        expected_sci = np.full((height, width), 2.0, dtype=np.float32)
        expected_var = np.full((height, width), 0.2, dtype=np.float32)
        expected_sci[1, 2] = np.nan
        expected_var[1, 2] = np.nan

        self.assertTrue(np.allclose(stack.sci[5], expected_sci, equal_nan=True))
        self.assertTrue(np.allclose(stack.var[5], expected_var, equal_nan=True))
        self.assertEqual(stack.times[5], 10.0)
        self.assertEqual(stack.zeroed_times[5], 10.0)

    def test_filter_image_stack_ph(self):
        """Test that we can filter an ImageStackPy"""
        num_times = 10
        height = 20
        width = 15

        times = np.arange(num_times) + 3.0
        sci = [np.full((height, width), float(i)) for i in range(num_times)]
        var = [np.full((height, width), 0.1 * float(i)) for i in range(num_times)]
        psfs = [np.array([[float(i)]]) for i in range(num_times)]

        stack = ImageStackPy(times, sci, var, psfs=psfs)
        self.assertEqual(stack.num_times, num_times)

        keep_inds = np.array([1, 3, 5, 6, 7, 8])
        keep_mask = np.full(num_times, False)
        keep_mask[keep_inds] = True
        stack.filter_images(keep_mask)
        self.assertEqual(stack.num_times, len(keep_inds))

        # Check that we saved the correct indices.
        for new_idx, org_idx in enumerate(keep_inds):
            self.assertTrue(np.allclose(stack.sci[new_idx], sci[org_idx]))
            self.assertTrue(np.allclose(stack.var[new_idx], var[org_idx]))
            self.assertTrue(np.allclose(stack.psfs[new_idx], psfs[org_idx]))

        # Check the times. Note the zeroed times shift because we removed index 0.
        self.assertTrue(np.allclose(stack.times, [4.0, 6.0, 8.0, 9.0, 10.0, 11.0]))
        self.assertTrue(np.allclose(stack.zeroed_times, [0.0, 2.0, 4.0, 5.0, 6.0, 7.0]))

    def test_get_matched_obstimes(self):
        obstimes = [1.0, 2.0, 3.0, 4.0, 6.0, 7.5, 9.0, 10.1]

        num_times = len(obstimes)
        height = 20
        width = 15
        sci = np.full((num_times, height, width), 1.0)
        var = np.full((num_times, height, width), 0.1)

        stack = ImageStackPy(obstimes, sci, var)

        query_times = [-1.0, 0.999999, 1.001, 1.1, 1.999, 6.001, 7.499, 7.5, 10.099999, 10.10001, 20.0]
        matched_inds = stack.get_matched_obstimes(query_times, threshold=0.01)
        expected = [-1, 0, 0, -1, 1, 4, 5, 5, 7, 7, -1]
        self.assertTrue(np.array_equal(matched_inds, expected))

    def test_make_fake_image_stack(self):
        """Test that we can create a fake ImageStackPy."""
        fake_times = np.arange(10)
        fake_stack = make_fake_image_stack(200, 300, fake_times)
        self.assertEqual(fake_stack.num_times, 10)
        self.assertEqual(fake_stack.width, 200)
        self.assertEqual(fake_stack.height, 300)
        self.assertEqual(fake_stack.npixels, 200 * 300)
        self.assertEqual(fake_stack.total_pixels, 10 * 200 * 300)
        self.assertEqual(fake_stack.num_masked_pixels(), 0)
        self.assertEqual(len(fake_stack.sci), 10)
        self.assertEqual(len(fake_stack.var), 10)
        for idx in range(10):
            self.assertEqual(fake_stack.sci[idx].shape, (300, 200))
            self.assertEqual(fake_stack.var[idx].shape, (300, 200))
            self.assertTrue(len(np.unique(fake_stack.sci[idx])) > 1)
            self.assertTrue(np.allclose(fake_stack.var[idx], 4.0))
        self.assertEqual(len(fake_stack.psfs), 10)

    def test_image_stack_add_fake_object(self):
        """Test that we can inset a fake object into an ImageStackPy."""
        num_times = 5
        height = 200
        width = 300

        fake_times = np.arange(num_times)
        sci = np.full((num_times, height, width), 0.0)
        var = np.full((num_times, height, width), 1.0)
        psfs = [PSF.from_gaussian(0.5) for i in range(num_times)]
        fake_stack = ImageStackPy(fake_times, sci, var, psfs=psfs)

        image_stack_add_fake_object(fake_stack, 50, 60, 1.0, 2.0, 100.0)
        for t_idx, t_val in enumerate(fake_times):
            # Check that we receive a signal at the correct location that
            # is non-zero but less than the flux (due to the PSF) at each time step.
            px = int(50 + t_val + 0.5)
            py = int(60 + 2.0 * t_val + 0.5)
            self.assertGreater(fake_stack.sci[t_idx][py, px], 50.0)
            self.assertLess(fake_stack.sci[t_idx][py, px], 100.0)

            # Far away from the object, the signal should be zero.
            self.assertAlmostEqual(fake_stack.sci[t_idx][30, 40], 0.0)

    def test_get_masked_fractions(self):
        """Test that we can compute the fraction of pixels masked at each index."""
        num_times = 20
        height = 20
        width = 30
        stack = ImageStackPy()

        # Append images with an increasing fraction of masked pixels
        for idx in range(num_times):
            sci = np.full((height, width), float(idx))
            var = np.full((height, width), 0.1 * float(idx))
            msk = np.zeros((height, width))
            msk[0:idx, :] = 1
            stack.append_image(idx / 20.0, sci, var, mask=msk)

        fractions = stack.get_masked_fractions()
        self.assertTrue(np.allclose(fractions, np.arange(num_times) / float(num_times)))

        # Filter out images with above a given masked fraction.
        stack.filter_images(fractions < 0.5)
        times_remaining = int(num_times / 2)
        self.assertEqual(stack.num_times, times_remaining)
        self.assertTrue(np.allclose(stack.times, np.arange(times_remaining) / float(num_times)))

    def test_mask_by_science_bounds(self):
        """Test that we can mask pixels by their science values."""
        height = 20
        width = 30

        times = [0.0]
        sci = [np.arange(height * width).reshape((height, width)).astype(np.float32)]
        var = [np.full((height, width), 0.1, dtype=np.float32)]

        stack = ImageStackPy(times, sci, var)
        fractions = stack.get_masked_fractions()
        self.assertEqual(fractions[0], 0.0)

        # Mask out everything outside [10.0, 110.0].
        stack.mask_by_science_bounds(min_val=10.0, max_val=110.0)
        fractions = stack.get_masked_fractions()
        self.assertAlmostEqual(fractions[0], 499.0 / 600.0)

        is_unmasked = ~np.isnan(stack.sci[0])
        self.assertTrue(np.all(stack.sci[0][is_unmasked] >= 10.0))
        self.assertTrue(np.all(stack.sci[0][is_unmasked] <= 110.0))

    def test_mask_by_variance_bounds(self):
        """Test that we can mask pixels by their science values."""
        height = 20
        width = 30

        times = [0.0]
        sci = [np.arange(height * width).reshape((height, width)).astype(np.float32)]
        var = [np.arange(height * width).reshape((height, width)).astype(np.float32) - 10.0]

        stack = ImageStackPy(times, sci, var)
        fractions = stack.get_masked_fractions()
        self.assertEqual(fractions[0], 0.0)

        # Mask out everything with a variance <= 0.0 or above 200.0)
        stack.mask_by_variance_bounds(max_val=200.0)
        fractions = stack.get_masked_fractions()
        self.assertAlmostEqual(fractions[0], 400.0 / 600.0)

        is_unmasked = ~np.isnan(stack.var[0])
        self.assertTrue(np.all(stack.var[0][is_unmasked] > 0.0))
        self.assertTrue(np.all(stack.var[0][is_unmasked] <= 200.0))

    def test_image_stack_py_validate(self):
        """Tests that we can validate an ImageStackPy."""
        # Turn off the warnings for this test.
        logging.disable(logging.CRITICAL)

        # Start with a valid ImageStack.
        num_times = 5
        width = 3
        height = 4
        fake_times = np.arange(num_times)
        fake_sci = [90.0 * np.random.random((height, width)) + 10.0 for _ in range(num_times)]
        fake_var = [0.49 * np.random.random((height, width)) + 0.01 for _ in range(num_times)]
        fake_msk = [np.zeros((height, width)) for _ in range(num_times)]
        im_stack = ImageStackPy(fake_times, fake_sci, fake_var)

        self.assertEqual(im_stack.num_times, 5)
        self.assertTrue(im_stack.validate())

        # Too high flux.
        fake_sci2 = np.full_like(fake_sci, 1.0)
        fake_sci2[1][2][1] = 1e9
        im_stack = ImageStackPy(fake_times, fake_sci2, fake_var, fake_msk)
        self.assertFalse(im_stack.validate())

        # Too low flux.
        fake_sci2[1][2][1] = -1e9
        im_stack = ImageStackPy(fake_times, fake_sci2, fake_var, fake_msk)
        self.assertFalse(im_stack.validate())

        # Too high variance.
        fake_var2 = np.full_like(fake_var, 1.0)
        fake_var2[1][2][1] = 1e9
        im_stack = ImageStackPy(fake_times, fake_sci, fake_var2, fake_msk)
        self.assertFalse(im_stack.validate())

        # Too low variance.
        fake_var2[1][2][1] = -1e9
        im_stack = ImageStackPy(fake_times, fake_sci, fake_var2, fake_msk)
        self.assertFalse(im_stack.validate())

        # Too many masked pixels in an image.
        fake_msk2 = np.full_like(fake_msk, 0)
        fake_msk2[1, :, :] = 1
        im_stack = ImageStackPy(fake_times, fake_sci, fake_var, fake_msk2)
        self.assertFalse(im_stack.validate())

        # Re-enable warnings.
        logging.disable(logging.NOTSET)


if __name__ == "__main__":
    unittest.main()
