import math
import numpy as np
import os
import tempfile
import unittest

from kbmod.search import (
    HAS_GPU,
    KB_NO_DATA,
    PSF,
    RawImage,
    create_median_image,
    create_summed_image,
    create_mean_image,
    pixel_value_valid,
)


class test_RawImage(unittest.TestCase):
    def setUp(self, width=10, height=12):
        self.width = width
        self.height = height

        # self.const_arr =  10.0 * np.ones(height, width, dtype=np.single)
        self.array = np.arange(0, width * height, dtype=np.single).reshape(height, width)

        self.masked_array = 10.0 * np.ones((height, width), dtype=np.single)
        self.masked_array[5, 6] = 0.1
        self.masked_array[5, 7] = KB_NO_DATA
        self.masked_array[3, 1] = 100.0
        self.masked_array[4, 4] = KB_NO_DATA
        self.masked_array[5, 5] = 100.0

    def test_create(self):
        """Test RawImage constructors."""
        # Default constructor
        img = RawImage()
        self.assertEqual(img.width, 0)
        self.assertEqual(img.height, 0)
        self.assertEqual(img.obstime, -1.0)

        # from NumPy arrays
        img = RawImage(img=self.array, obs_time=10.0)
        self.assertEqual(img.image.shape, (self.height, self.width))
        self.assertEqual(img.obstime, 10.0)
        self.assertEqual(img.npixels, self.width * self.height)
        self.assertTrue((img.image == self.array).all())

        img2 = RawImage(img=self.array)
        self.assertTrue((img2.image == img.image).all())
        self.assertEqual(img2.obstime, -1.0)

        # from dimensions
        img = RawImage(self.width, self.height)
        self.assertEqual(img.image.shape, (self.height, self.width))
        self.assertEqual(img.obstime, -1.0)
        self.assertTrue((img.image == 0).all())

        # dimensions and optional values
        img = RawImage(self.height, self.width, 10)
        self.assertTrue((img.image == 10).all())

        img = RawImage(self.height, self.width, 10, 12.5)
        self.assertTrue((img.image == 10).all())
        self.assertEqual(img.obstime, 12.5)

        img = RawImage(self.height, self.width, value=7.5, obs_time=12.5)
        self.assertTrue((img.image == 7.5).all())
        self.assertEqual(img.obstime, 12.5)

        # copy constructor, set the old image to all zeros and change the time.
        img = RawImage(img=self.array, obs_time=10.0)
        img2 = RawImage(img)
        img.set_all(0.0)
        img.obstime = 1.0
        self.assertTrue((img2.image == self.array).all())
        self.assertEqual(img2.obstime, 10.0)

    def test_pixel_getters(self):
        """Test RawImage masked pixel value getters"""
        img = RawImage(img=self.array, obs_time=10.0)
        self.assertFalse(pixel_value_valid(img.get_pixel(-1, 5)))
        self.assertFalse(pixel_value_valid(img.get_pixel(5, self.width)))
        self.assertFalse(pixel_value_valid(img.get_pixel(5, -1)))
        self.assertFalse(pixel_value_valid(img.get_pixel(self.height, 5)))

    def test_validity_checker(self):
        img = RawImage(img=np.array([[0, 0], [0, 0]]).astype(np.float32), obs_time=10.0)
        self.assertTrue(img.pixel_has_data(0, 0))

        img.set_pixel(0, 0, np.nan)
        self.assertFalse(img.pixel_has_data(0, 0))

        img.set_pixel(0, 0, np.inf)
        self.assertFalse(img.pixel_has_data(0, 0))

        img.set_pixel(0, 0, -np.inf)
        self.assertFalse(img.pixel_has_data(0, 0))

        img.mask_pixel(0, 0)
        self.assertFalse(img.pixel_has_data(0, 0))

    def test_interpolated_add(self):
        """Test that we can add values to the pixel."""
        img = RawImage(img=self.array, obs_time=10.0)

        # Get the original value using (r, c) lookup.
        org_val17 = img.get_pixel(1, 7)

        # Interpolated add uses the cartesian coordinates (x, y)
        img.interpolated_add(7, 1, 10.0)
        self.assertLess(img.get_pixel(1, 7), org_val17 + 10.0)
        self.assertGreater(img.get_pixel(1, 7), org_val17 + 2.0)

    def test_approx_equal(self):
        """Test RawImage pixel value setters."""
        img = RawImage(img=self.array, obs_time=10.0)

        # This test is testing L^\infy norm closeness. Eigen isApprox uses L2
        # norm closeness.
        img2 = RawImage(img)
        img2.imref += 0.0001
        self.assertTrue(img.l2_allclose(img2, 0.01))

        # Add a single masked entry.
        img.mask_pixel(5, 7)
        self.assertFalse(img.l2_allclose(img2, 0.01))

        img2.mask_pixel(5, 7)
        self.assertTrue(img.l2_allclose(img2, 0.01))

        # Add a second masked entry to image 2.
        img2.mask_pixel(7, 7)
        self.assertFalse(img.l2_allclose(img2, 0.01))

        img.mask_pixel(7, 7)
        self.assertTrue(img.l2_allclose(img2, 0.01))

        # Add some noise to mess up an observation.
        img2.set_pixel(1, 3, 13.1)
        self.assertFalse(img.l2_allclose(img2, 0.01))

        # test set_all
        img.set_all(15.0)
        self.assertTrue((img.image == 15).all())

    def test_get_bounds(self):
        """Test RawImage masked min/max bounds."""
        img = RawImage(self.masked_array)
        lower, upper = img.compute_bounds()
        self.assertAlmostEqual(lower, 0.1, delta=1e-6)
        self.assertAlmostEqual(upper, 100.0, delta=1e-6)

        # Insert a NaN and make sure that does not mess up the computation.
        img.set_pixel(2, 3, math.nan)
        img.set_pixel(3, 2, np.nan)
        lower, upper = img.compute_bounds()
        self.assertAlmostEqual(lower, 0.1, delta=1e-6)
        self.assertAlmostEqual(upper, 100.0, delta=1e-6)

    def test_replace_masked_values(self):
        img2 = RawImage(np.copy(self.masked_array))
        img2.replace_masked_values(0.0)

        for row in range(img2.height):
            for col in range(img2.width):
                if pixel_value_valid(self.masked_array[row, col]):
                    self.assertEqual(
                        self.masked_array[row, col],
                        img2.get_pixel(row, col),
                    )
                else:
                    self.assertEqual(img2.get_pixel(row, col), 0.0)

    def test_find_peak(self):
        "Test RawImage find_peak"
        img = RawImage(self.masked_array)
        idx = img.find_peak(False)
        self.assertEqual(idx.i, 5)
        self.assertEqual(idx.j, 5)

        # We found the peak furthest to the center.
        idx = img.find_peak(True)
        self.assertEqual(idx.i, 3)
        self.assertEqual(idx.j, 1)

        # We are okay when the data includes NaNs.
        img.set_pixel(2, 3, math.nan)
        img.set_pixel(3, 2, np.nan)
        idx = img.find_peak(False)
        self.assertEqual(idx.i, 5)
        self.assertEqual(idx.j, 5)

    def test_find_central_moments(self):
        """Test RawImage central moments."""
        img = RawImage(5, 5, value=0.1)

        # Try something mostly symmetric and centered.
        img.set_pixel(2, 2, 10.0)
        img.set_pixel(2, 1, 5.0)
        img.set_pixel(1, 2, 5.0)
        img.set_pixel(2, 3, 5.0)
        img.set_pixel(3, 2, 5.0)

        img_mom = img.find_central_moments()
        self.assertAlmostEqual(img_mom.m00, 1.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m01, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m10, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m11, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m02, 0.3322, delta=1e-4)
        self.assertAlmostEqual(img_mom.m20, 0.3322, delta=1e-4)

        # Try something flat symmetric and centered.
        img.set_all(2.0)
        img_mom = img.find_central_moments()

        self.assertAlmostEqual(img_mom.m00, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m01, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m10, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m11, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m02, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m20, 0.0, delta=1e-4)

        # Try something with a few non-symmetric peaks.
        img.set_all(0.4)
        img.set_pixel(2, 2, 5.0)
        img.set_pixel(0, 1, 5.0)
        img.set_pixel(3, 3, 10.0)
        img.set_pixel(0, 3, 0.2)
        img_mom = img.find_central_moments()

        self.assertAlmostEqual(img_mom.m00, 1.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m01, 0.20339, delta=1e-4)
        self.assertAlmostEqual(img_mom.m10, 0.03390, delta=1e-4)
        self.assertAlmostEqual(img_mom.m11, 0.81356, delta=1e-4)
        self.assertAlmostEqual(img_mom.m02, 1.01695, delta=1e-4)
        self.assertAlmostEqual(img_mom.m20, 1.57627, delta=1e-4)

        # Check that nothing fails with NaNs.
        img.set_pixel(2, 3, math.nan)
        img.set_pixel(3, 2, np.nan)
        img_mom = img.find_central_moments()

    def convolve_psf_identity(self, device):
        psf_data = np.zeros((3, 3), dtype=np.single)
        psf_data[1, 1] = 1.0
        p = PSF(psf_data)

        img = RawImage(self.array)

        if device.upper() == "CPU":
            img.convolve_cpu(p)
        elif device.upper() == "GPU":
            img.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        self.assertTrue(np.allclose(self.array, img.image, 0.0001))

    def test_convolve_psf_identity_cpu(self):
        """Test convolution with a identity kernel on CPU"""
        self.convolve_psf_identity("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_identity_gpu(self):
        """Test convolution with a identity kernel on GPU"""
        self.convolve_psf_identity("GPU")

    def convolve_psf_mask(self, device):
        p = PSF(1.0)

        # Mask out three pixels.
        img = RawImage(self.array)
        img.mask_pixel(0, 3)
        img.mask_pixel(5, 6)
        img.mask_pixel(5, 7)

        if device.upper() == "CPU":
            img.convolve_cpu(p)
        elif device.upper() == "GPU":
            img.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(img.pixel_has_data(y, x))
                else:
                    self.assertTrue(img.pixel_has_data(y, x))

    def test_convolve_psf_mask_cpu(self):
        """Test masked convolution with a identity kernel on CPU"""
        self.convolve_psf_mask("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_mask_gpu(self):
        """Test masked convolution with a identity kernel on GPU"""
        self.convolve_psf_mask("GPU")

    def convolve_psf_nan(self, device):
        p = PSF(1.0)

        # Mask out three pixels.
        img = RawImage(self.array)
        img.set_pixel(0, 3, math.nan)
        img.set_pixel(5, 6, np.nan)
        img.set_pixel(5, 7, np.nan)

        if device.upper() == "CPU":
            img.convolve_cpu(p)
        elif device.upper() == "GPU":
            img.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        # Check that the same pixels are NaN (we ignore those pixels).
        for y in range(self.height):
            for x in range(self.width):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertTrue(math.isnan(img.get_pixel(y, x)))
                else:
                    self.assertFalse(math.isnan(img.get_pixel(y, x)))

    def test_convolve_psf_nan_cpu(self):
        self.convolve_psf_nan("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_nan_gpu(self):
        self.convolve_psf_nan("GPU")

    # confused, sort out later
    def convolve_psf_average(self, device):
        # Mask out a single pixel.
        img = RawImage(self.array)
        img.mask_pixel(4, 6)

        # Set up a simple "averaging" psf to convolve.
        psf_data = np.zeros((5, 5), dtype=np.single)
        psf_data[1:4, 1:4] = 0.1111111
        p = PSF(psf_data)
        self.assertAlmostEqual(p.get_sum(), 1.0, delta=0.00001)

        img2 = RawImage(img)
        if device.upper() == "CPU":
            img2.convolve_cpu(p)
        elif device.upper() == "GPU":
            img2.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        for x in range(self.width):
            for y in range(self.height):
                # Compute the weighted average around (x, y)
                # in the original image.
                running_sum = 0.0
                count = 0.0
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        value = img.get_pixel(y + j, x + i)
                        psf_value = 0.1111111
                        if i == -2 or i == 2 or j == -2 or j == 2:
                            psf_value = 0.0

                        if pixel_value_valid(value):
                            running_sum += psf_value * value
                            count += psf_value
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                if x == 6 and y == 4:
                    self.assertFalse(img2.pixel_has_data(y, x))
                else:
                    self.assertAlmostEqual(img2.get_pixel(y, x), ave, delta=0.001)

    def test_convolve_psf_average(self):
        """Test convolution on CPU produces expected values."""
        self.convolve_psf_average("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_average_gpu(self):
        """Test convolution on GPU produces expected values."""
        self.convolve_psf_average("GPU")

    def convolve_psf_orientation_cpu(self, device):
        """Test convolution on CPU with a non-symmetric PSF"""
        img = RawImage(self.array.copy())

        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = PSF(np.array(psf_data))

        img2 = RawImage(img)
        if device.upper() == "CPU":
            img2.convolve_cpu(p)
        elif device.upper() == "GPU":
            img2.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        for x in range(img.width):
            for y in range(img.height):
                running_sum = 0.5 * img.get_pixel(y, x)
                count = 0.5
                if img.pixel_has_data(y, x + 1):
                    running_sum += 0.4 * img.get_pixel(y, x + 1)
                    count += 0.4
                if img.pixel_has_data(y + 1, x):
                    running_sum += 0.1 * img.get_pixel(y + 1, x)
                    count += 0.1
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                self.assertAlmostEqual(img2.get_pixel(y, x), ave, delta=0.001)

    def test_convolve_psf_orientation_cpu(self):
        """Test convolution on CPU with a non-symmetric PSF"""
        self.convolve_psf_orientation_cpu("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_orientation_gpu(self):
        """Test convolution on GPU with a non-symmetric PSF"""
        self.convolve_psf_orientation_cpu("GPU")

    # Stamp as is tested here and as it's used in StackSearch are heaven and earth
    # TODO: Add proper tests
    def test_make_stamp(self):
        """Test stamp creation."""
        img = RawImage(self.array)
        stamp = img.create_stamp(2.5, 2.5, 2, False)
        self.assertEqual(stamp.image.shape, (5, 5))
        self.assertTrue((stamp.image == self.array[0:5, 0:5]).all())

        # Test a stamp that is not at the corner.
        stamp = img.create_stamp(8.5, 5.5, 1, False)
        self.assertEqual(stamp.image.shape, (3, 3))
        self.assertTrue((stamp.image == self.array[4:7, 7:10]).all())

        # Test a stamp with masked pixels.
        img2 = RawImage(self.masked_array)
        stamp = img2.create_stamp(7.5, 5.5, 1, True)
        self.assertEqual(stamp.image.shape, (3, 3))
        stamp2 = RawImage(img2.image[4:7, 6:9])
        self.assertTrue(stamp.l2_allclose(stamp2, 0.01))

        # Test a stamp with masked pixels and replacement.
        stamp = img2.create_stamp(7.5, 5.5, 2, False)
        self.assertEqual(stamp.image.shape, (5, 5))
        expected_stamp = RawImage(np.copy(self.masked_array[3:8, 5:10]))
        expected_stamp.replace_masked_values(0.0)
        self.assertTrue((stamp.image == expected_stamp.image).all())

        # Test a stamp that goes out of bounds.
        stamp = img.create_stamp(0.5, 11.5, 1, False)
        expected = np.array([[0.0, 100.0, 101.0], [0.0, 110.0, 111.0], [0.0, 0.0, 0.0]])
        self.assertTrue((stamp.image == expected).all())

        # Test a stamp that is completely out of bounds.
        stamp = img.create_stamp(20.5, 20.5, 1, False)
        expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertTrue((stamp.image == expected).all())

        # Test a stamp that overlaps at a single corner pixel.
        stamp = img.create_stamp(-1.5, -1.5, 1, True)
        for row in range(3):
            for col in range(3):
                if row == 2 and col == 2:
                    self.assertEqual(stamp.image[row][col], 0.0)
                else:
                    self.assertFalse(stamp.pixel_has_data(row, col))

    def test_create_median_image(self):
        """Tests median image coaddition."""
        arrs = np.array(
            [
                [[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]],
                [[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]],
                [[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]],
            ],
            dtype=np.single,
        )
        imgs = list(map(RawImage, arrs))

        median_image = create_median_image(imgs)

        expected = np.median(arrs, axis=0)
        self.assertEqual(median_image.width, 2)
        self.assertEqual(median_image.height, 3)
        self.assertTrue(np.allclose(median_image.image, expected, atol=1e-6))

        # Apply masks to images 1 and 3.
        imgs[0].apply_mask(1, RawImage(np.array([[0, 1], [0, 1], [0, 1]], dtype=np.single)))
        imgs[2].apply_mask(1, RawImage(np.array([[0, 0], [1, 1], [1, 0]], dtype=np.single)))

        median_image = create_median_image(imgs)

        expected = np.array([[0, -1], [1.5, 3.5], [2.35, 3.15]], dtype=np.single)
        self.assertEqual(median_image.width, 2)
        self.assertEqual(median_image.height, 3)
        self.assertTrue(np.allclose(median_image.image, expected, atol=1e-6))

        # More median image tests
        arrs = np.array(
            [
                [[1.0, -1.0], [-1.0, 1.0], [1.0, 0.1]],
                [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
                [[3.0, -2.0], [-2.0, 5.0], [4.0, 0.3]],
                [[4.0, 3.0], [3.0, 6.0], [5.0, 0.1]],
                [[5.0, -3.0], [-3.0, 7.0], [7.0, 0.0]],
                [[6.0, 2.0], [2.0, 4.0], [6.0, 0.1]],
                [[7.0, 3.0], [3.0, 3.0], [3.0, 0.0]],
            ],
            dtype=np.single,
        )

        masks = np.array(
            [
                np.array([[0, 0], [1, 1], [0, 0]]),
                np.array([[0, 0], [1, 1], [1, 0]]),
                np.array([[0, 0], [0, 1], [0, 0]]),
                np.array([[0, 0], [0, 1], [0, 0]]),
                np.array([[0, 1], [0, 1], [0, 0]]),
                np.array([[0, 1], [1, 1], [0, 0]]),
                np.array([[0, 0], [1, 1], [0, 0]]),
            ],
            dtype=np.single,
        )

        imgs = list(map(RawImage, arrs))
        for img, mask in zip(imgs, masks):
            img.apply_mask(1, RawImage(mask))

        median_image = create_median_image(imgs)
        expected = np.array([[4, 0], [-2, 0], [4.5, 0.1]], dtype=np.single)
        self.assertEqual(median_image.width, 2)
        self.assertEqual(median_image.height, 3)
        self.assertTrue(np.allclose(median_image.image, expected, atol=1e-6))

    def test_create_summed_image(self):
        arrs = np.array(
            [
                [[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]],
                [[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]],
                [[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]],
            ],
            dtype=np.single,
        )
        imgs = list(map(RawImage, arrs))

        summed_image = create_summed_image(imgs)

        expected = arrs.sum(axis=0)
        self.assertEqual(summed_image.width, 2)
        self.assertEqual(summed_image.height, 3)
        self.assertTrue(np.allclose(expected, summed_image.image, atol=1e-6))

        # Apply masks to images 1 and 3.
        imgs[0].apply_mask(1, RawImage(np.array([[0, 1], [0, 1], [0, 1]], dtype=np.single)))
        imgs[2].apply_mask(1, RawImage(np.array([[0, 0], [1, 1], [1, 0]], dtype=np.single)))

        summed_image = create_summed_image(imgs)

        expected = np.array([[0, -2], [3, 3.5], [4.7, 6.3]], dtype=np.single)
        self.assertEqual(summed_image.width, 2)
        self.assertEqual(summed_image.height, 3)
        self.assertTrue(np.allclose(expected, summed_image.image, atol=1e-6))

    def test_create_mean_image(self):
        arrs = np.array(
            [
                [[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]],
                [[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]],
                [[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]],
            ],
            dtype=np.single,
        )
        imgs = list(map(RawImage, arrs))

        mean_image = create_mean_image(imgs)

        expected = arrs.mean(axis=0)
        self.assertEqual(mean_image.width, 2)
        self.assertEqual(mean_image.height, 3)
        self.assertTrue(np.allclose(mean_image.image, expected, atol=1e-6))

        # Apply masks to images 1, 2, and 3.
        masks = np.array(
            [[[0, 1], [0, 1], [0, 1]], [[0, 0], [0, 0], [0, 1]], [[0, 0], [1, 1], [1, 1]]], dtype=np.single
        )
        for img, mask in zip(imgs, masks):
            img.apply_mask(1, RawImage(mask))

        mean_image = create_mean_image(imgs)

        expected = np.array([[0, -1], [1.5, 3.5], [2.35, 0]], dtype=np.single)
        self.assertEqual(mean_image.width, 2)
        self.assertEqual(mean_image.height, 3)
        self.assertTrue(np.allclose(mean_image.image, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
