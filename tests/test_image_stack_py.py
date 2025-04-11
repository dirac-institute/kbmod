import astropy.wcs
import numpy as np
import unittest

from kbmod.core.image_stack_py import (
    make_fake_image_stack,
    image_stack_add_fake_object,
    ImageStackPy,
)
from kbmod.core.psf import PSF


class test_image_stack_py(unittest.TestCase):
    def test_create_image_stack_py(self):
        """Test that we can create an ImageStackPy"""
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

        self.assertTrue(np.all(stack.sci == 1.0))
        self.assertTrue(np.all(stack.var == 0.1))
        self.assertEqual(len(stack.psfs), num_times)

        # Test that we fail with bad input.
        self.assertRaises(ValueError, ImageStackPy, np.array([]), sci, var)
        self.assertRaises(ValueError, ImageStackPy, None, sci, var)
        self.assertRaises(ValueError, ImageStackPy, times, np.array([1.0]), var)
        self.assertRaises(ValueError, ImageStackPy, times, sci, np.array([1.0]))

    def test_make_empty_image_stack(self):
        """Test that we can create an empty ImageStackPy"""
        num_times = 5
        height = 6
        width = 4

        stack = ImageStackPy.make_empty(num_times, height, width)
        self.assertEqual(stack.num_times, num_times)
        self.assertEqual(stack.width, width)
        self.assertEqual(stack.height, height)
        self.assertEqual(stack.npixels, height * width)
        self.assertEqual(stack.total_pixels, num_times * height * width)
        self.assertEqual(stack.num_masked_pixels(), 0)
        self.assertTrue(np.all(stack.times == 0.0))
        self.assertTrue(np.all(stack.zeroed_times == 0.0))
        self.assertTrue(np.all(stack.sci == 0.0))
        self.assertTrue(np.all(stack.var == 0.0))
        self.assertTrue(np.all(stack.mask == False))

        # Test that we can insert a layer.
        fake_sci = 5.0 * np.ones((height, width), dtype=np.single)
        fake_var = np.ones((height, width), dtype=np.single)
        fake_mask = np.zeros((height, width))
        fake_mask[2, 1] = 1
        stack.set_images_at_time(3, fake_sci, fake_var, mask=fake_mask)

        # Compute the expected (masked) values.
        expected_sci = 5.0 * np.ones((height, width), dtype=np.single)
        expected_var = np.ones((height, width), dtype=np.single)
        expected_mask = np.full((height, width), False, dtype=bool)
        expected_mask[2, 1] = True
        expected_sci[2, 1] = np.nan
        expected_var[2, 1] = np.nan

        # The inserted layer should have the correct values.
        self.assertTrue(np.array_equal(stack.sci[3], expected_sci, equal_nan=True))
        self.assertTrue(np.array_equal(stack.var[3], expected_var, equal_nan=True))
        self.assertTrue(np.array_equal(stack.mask[3], expected_mask))

        # All other times are unchanged.
        for i in range(num_times):
            if i != 3:
                self.assertTrue(np.all(stack.sci[i] == 0.0))
                self.assertTrue(np.all(stack.var[i] == 0.0))
                self.assertTrue(np.all(stack.mask[i] == False))

    def test_create_image_stack_py_masked(self):
        """Test that we can create an ImageStackPy"""
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
        mask_mask = mask > 0

        stack = ImageStackPy(times, sci, var, mask=mask)
        self.assertEqual(stack.num_times, num_times)
        self.assertEqual(stack.width, width)
        self.assertEqual(stack.height, height)
        self.assertEqual(stack.npixels, height * width)
        self.assertEqual(stack.total_pixels, num_times * height * width)
        self.assertEqual(stack.num_masked_pixels(), 3)

        self.assertTrue(np.allclose(stack.times, times))
        self.assertTrue(np.allclose(stack.zeroed_times, np.arange(num_times)))

        self.assertTrue(np.all(stack.sci[~mask_mask] == 1.0))
        self.assertTrue(np.all(np.isnan(stack.sci[mask_mask])))

        self.assertTrue(np.all(stack.var[~mask_mask] == 0.1))
        self.assertTrue(np.all(np.isnan(stack.var[mask_mask])))

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

    def test_pixel_to_from_world(self):
        num_times = 10
        height = 500
        width = 700
        times = np.arange(num_times)

        # Create a fake WCS for testing.
        wcs_dict = {
            "WCSAXES": 2,
            "CDELT1": 0.01,
            "CDELT2": 0.01,
            "CTYPE1": "RA---TAN-SIP",
            "CTYPE2": "DEC--TAN-SIP",
            "CRVAL1": 200.5,
            "CRVAL2": -7.5,
            "CRPIX1": height / 2.0,
            "CRPIX2": width / 2.0,
            "CTYPE1A": "LINEAR  ",
            "CTYPE2A": "LINEAR  ",
            "CUNIT1A": "PIXEL   ",
            "CUNIT2A": "PIXEL   ",
            "NAXIS1": height,
            "NAXIX2": width,
        }
        wcs = astropy.wcs.WCS(wcs_dict)
        wcs.array_shape = (height, width)
        wcs_list = [wcs if i % 2 == 0 else None for i in range(num_times)]

        # Create a fake image stack.
        sci = np.full((num_times, height, width), 1.0)
        var = np.full((num_times, height, width), 0.1)
        stack = ImageStackPy(times, sci, var, wcs=wcs_list)

        # Compute the pixel locations of the SkyCoords.
        query_ra = np.array([200.5, 200.55, 200.6])
        query_dec = np.array([-7.5, -7.55, -7.60])
        expected_x = np.array([249, 254, 259])
        expected_y = np.array([349, 344, 339])
        for idx in range(3):
            x_pos, y_pos = stack.world_to_pixel(0, query_ra[idx], query_dec[idx])
            self.assertAlmostEqual(x_pos, expected_x[idx], delta=0.2)
            self.assertAlmostEqual(y_pos, expected_y[idx], delta=0.2)

            ra, dec = stack.pixel_to_world(0, expected_x[idx], expected_y[idx])
            self.assertAlmostEqual(ra, query_ra[idx], delta=0.001)
            self.assertAlmostEqual(dec, query_dec[idx], delta=0.001)

        # We fail if the WCS is not set.
        with self.assertRaises(ValueError):
            stack.world_to_pixel(1, 200.5, -7.5)
        with self.assertRaises(ValueError):
            stack.pixel_to_world(1, 249, 349)

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
        self.assertEqual(fake_stack.sci.shape, (10, 300, 200))
        self.assertEqual(fake_stack.var.shape, (10, 300, 200))
        self.assertEqual(len(fake_stack.psfs), 10)

        self.assertTrue(len(np.unique(fake_stack.sci)) > 1)
        self.assertTrue(np.allclose(fake_stack.var, 4.0))

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
            self.assertGreater(fake_stack.sci[t_idx, py, px], 50.0)
            self.assertLess(fake_stack.sci[t_idx, py, px], 100.0)

            # Far away from the object, the signal should be zero.
            self.assertAlmostEqual(fake_stack.sci[t_idx, 30, 40], 0.0)


if __name__ == "__main__":
    unittest.main()
