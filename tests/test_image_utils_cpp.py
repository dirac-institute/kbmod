import math
import numpy as np
import unittest

from kbmod.core.psf import PSF
from kbmod.search import (
    HAS_GPU,
    KB_NO_DATA,
    convolve_image_cpu,
    convolve_image_gpu,
    pixel_value_valid,
)


class test_image_utils_cpp(unittest.TestCase):
    def setUp(self, width=10, height=12):
        self.width = width
        self.height = height
        self.array = np.arange(0, width * height, dtype=np.single).reshape(height, width)

    def test_convolve_psf_identity_cpu(self):
        """Test convolution with a identity kernel on CPU"""
        psf_data = np.zeros((3, 3), dtype=np.single)
        psf_data[1, 1] = 1.0

        # The convolution should be no-op.
        result = convolve_image_cpu(self.array, psf_data)
        self.assertTrue(np.allclose(self.array, result, 0.0001))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_identity_gpu(self):
        """Test convolution with a identity kernel on CPU"""
        psf_data = np.zeros((3, 3), dtype=np.single)
        psf_data[1, 1] = 1.0

        # The convolution should be no-op.
        result = convolve_image_gpu(self.array, psf_data)
        self.assertTrue(np.allclose(self.array, result, 0.0001))

    def test_convolve_psf_mask_cpu(self):
        """Test masked convolution with a identity kernel on CPU"""
        p = PSF.make_gaussian_kernel(1.0)

        # Mask out three pixels.
        self.array[0, 3] = KB_NO_DATA
        self.array[5, 6] = KB_NO_DATA
        self.array[5, 7] = KB_NO_DATA

        result = convolve_image_cpu(self.array, p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(np.isfinite(result[y, x]))
                else:
                    self.assertTrue(np.isfinite(result[y, x]))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_mask_gpu(self):
        """Test masked convolution with a identity kernel on CPU"""
        p = PSF.make_gaussian_kernel(1.0)

        # Mask out three pixels.
        self.array[0, 3] = KB_NO_DATA
        self.array[5, 6] = KB_NO_DATA
        self.array[5, 7] = KB_NO_DATA

        result = convolve_image_gpu(self.array, p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(np.isfinite(result[y, x]))
                else:
                    self.assertTrue(np.isfinite(result[y, x]))

    def test_convolve_psf_nan_cpu(self):
        p = PSF.make_gaussian_kernel(1.0)

        # Mask out three pixels with math and np nans.
        self.array[0, 3] = math.nan
        self.array[5, 6] = np.nan
        self.array[5, 7] = np.nan

        result = convolve_image_cpu(self.array, p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(np.isfinite(result[y, x]))
                else:
                    self.assertTrue(np.isfinite(result[y, x]))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_nan_gpu(self):
        p = PSF.make_gaussian_kernel(1.0)

        # Mask out three pixels with math and np nans.
        self.array[0, 3] = math.nan
        self.array[5, 6] = np.nan
        self.array[5, 7] = np.nan

        result = convolve_image_gpu(self.array, p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(np.isfinite(result[y, x]))
                else:
                    self.assertTrue(np.isfinite(result[y, x]))

    def test_convolve_psf_average_cpu(self):
        """Test convolution on CPU produces expected values."""
        # Mask out a single pixel.
        self.array[4, 6] = KB_NO_DATA

        # Set up a simple "averaging" psf to convolve.
        p = np.zeros((5, 5), dtype=np.single)
        p[1:4, 1:4] = 0.1111111

        result = convolve_image_cpu(self.array, p)

        for x in range(self.width):
            for y in range(self.height):
                # Compute the weighted average around (x, y) in the original image.
                running_sum = 0.0
                count = 0.0
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        psf_value = p[2 + i, 2 + j]

                        px = x + i
                        py = y + j
                        if py >= 0 and py < self.height and px >= 0 and px < self.width:
                            value = self.array[py, px]
                        else:
                            # Out of bounds. Don't count either the sum or the PSF.
                            psf_value = 0.0
                            value = 0.0

                        if pixel_value_valid(value):
                            running_sum += psf_value * value
                            count += psf_value
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                if x == 6 and y == 4:
                    self.assertFalse(np.isfinite(result[y, x]))
                else:
                    self.assertAlmostEqual(result[y, x], ave, delta=0.001)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_average_gpu(self):
        """Test convolution on GPU produces expected values."""
        # Mask out a single pixel.
        self.array[4, 6] = KB_NO_DATA

        # Set up a simple "averaging" psf to convolve.
        p = np.zeros((5, 5), dtype=np.single)
        p[1:4, 1:4] = 0.1111111

        result = convolve_image_gpu(self.array, p)

        for x in range(self.width):
            for y in range(self.height):
                # Compute the weighted average around (x, y) in the original image.
                running_sum = 0.0
                count = 0.0
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        psf_value = p[2 + i, 2 + j]

                        px = x + i
                        py = y + j
                        if py >= 0 and py < self.height and px >= 0 and px < self.width:
                            value = self.array[py, px]
                        else:
                            # Out of bounds. Don't count either the sum or the PSF.
                            psf_value = 0.0
                            value = 0.0

                        if pixel_value_valid(value):
                            running_sum += psf_value * value
                            count += psf_value
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                if x == 6 and y == 4:
                    self.assertFalse(np.isfinite(result[y, x]))
                else:
                    self.assertAlmostEqual(result[y, x], ave, delta=0.001)

    def test_convolve_psf_orientation_cpu(self):
        """Test convolution on CPU with a non-symmetric PSF"""
        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = np.array(psf_data)

        result = convolve_image_cpu(self.array, p)

        for x in range(self.width):
            for y in range(self.height):
                running_sum = 0.5 * self.array[y, x]
                count = 0.5
                if x + 1 < self.width:
                    running_sum += 0.4 * self.array[y, x + 1]
                    count += 0.4
                if y + 1 < self.height:
                    running_sum += 0.1 * self.array[y + 1, x]
                    count += 0.1
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                self.assertAlmostEqual(result[y, x], ave, delta=0.001)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_orientation_gpu(self):
        """Test convolution on GPU with a non-symmetric PSF"""
        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = np.array(psf_data)

        result = convolve_image_gpu(self.array, p)

        for x in range(self.width):
            for y in range(self.height):
                running_sum = 0.5 * self.array[y, x]
                count = 0.5
                if x + 1 < self.width:
                    running_sum += 0.4 * self.array[y, x + 1]
                    count += 0.4
                if y + 1 < self.height:
                    running_sum += 0.1 * self.array[y + 1, x]
                    count += 0.1
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                self.assertAlmostEqual(result[y, x], ave, delta=0.001)


if __name__ == "__main__":
    unittest.main()
