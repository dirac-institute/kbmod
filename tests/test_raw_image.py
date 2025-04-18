import math
import numpy as np
import unittest

from kbmod.core.psf import PSF
from kbmod.image_utils import image_allclose
from kbmod.search import (
    HAS_GPU,
    KB_NO_DATA,
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

        # from NumPy arrays
        img = RawImage(img=self.array)
        self.assertEqual(img.image.shape, (self.height, self.width))
        self.assertEqual(img.npixels, self.width * self.height)
        self.assertTrue((img.image == self.array).all())

        img2 = RawImage(img=self.array)
        self.assertTrue((img2.image == img.image).all())

        # from dimensions
        img = RawImage(self.width, self.height)
        self.assertEqual(img.image.shape, (self.height, self.width))
        self.assertTrue((img.image == 0).all())

        # dimensions and optional values
        img = RawImage(self.height, self.width, 10)
        self.assertTrue((img.image == 10).all())

        img = RawImage(self.height, self.width, 10)
        self.assertTrue((img.image == 10).all())

        img = RawImage(self.height, self.width, value=7.5)
        self.assertTrue((img.image == 7.5).all())

        # copy constructor, set the old image to all zeros and change the time.
        img = RawImage(img=self.array)
        img2 = RawImage(img)
        img.set_all(0.0)
        self.assertTrue((img2.image == self.array).all())

    def test_pixel_getters(self):
        """Test RawImage masked pixel value getters"""
        img = RawImage(img=self.array)
        self.assertFalse(pixel_value_valid(img.get_pixel(-1, 5)))
        self.assertFalse(pixel_value_valid(img.get_pixel(5, self.width)))
        self.assertFalse(pixel_value_valid(img.get_pixel(5, -1)))
        self.assertFalse(pixel_value_valid(img.get_pixel(self.height, 5)))

    def test_contains(self):
        img = RawImage(img=self.array)
        self.assertTrue(img.contains_index(0, 0))
        self.assertTrue(img.contains_index(1, 2))
        self.assertFalse(img.contains_index(1, -1))
        self.assertFalse(img.contains_index(-1, 1))
        self.assertFalse(img.contains_index(1, self.width))
        self.assertFalse(img.contains_index(self.height, 1))

        # Works with floats
        self.assertTrue(img.contains_point(0.0, 0.0))
        self.assertTrue(img.contains_point(1.0, 2.0))
        self.assertFalse(img.contains_point(1.0, -1.0))
        self.assertFalse(img.contains_point(-1.0, 1.0))
        self.assertFalse(img.contains_point(self.width + 1e-4, 1.0))
        self.assertFalse(img.contains_point(1.0, self.height + 1e-4))

    def test_validity_checker(self):
        img = RawImage(img=np.array([[0, 0], [0, 0]]).astype(np.float32))
        self.assertTrue(img.pixel_has_data(0, 0))

        img.set_pixel(0, 0, np.nan)
        self.assertFalse(img.pixel_has_data(0, 0))

        img.set_pixel(0, 0, np.inf)
        self.assertFalse(img.pixel_has_data(0, 0))

        img.set_pixel(0, 0, -np.inf)
        self.assertFalse(img.pixel_has_data(0, 0))

        img.mask_pixel(0, 0)
        self.assertFalse(img.pixel_has_data(0, 0))

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

    def test_convolve_psf_identity(self):
        psf_data = np.zeros((3, 3), dtype=np.single)
        psf_data[1, 1] = 1.0

        img = RawImage(self.array)
        img.convolve(psf_data)
        self.assertTrue(np.allclose(self.array, img.image, 0.0001))

    def test_convolve_psf_mask(self):
        p = PSF.make_gaussian_kernel(1.0)

        # Mask out three pixels.
        img = RawImage(self.array)
        img.mask_pixel(0, 3)
        img.mask_pixel(5, 6)
        img.mask_pixel(5, 7)

        img.convolve(p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(img.pixel_has_data(y, x))
                else:
                    self.assertTrue(img.pixel_has_data(y, x))

    def test_convolve_psf_nan(self):
        p = PSF.make_gaussian_kernel(1.0)

        # Mask out three pixels.
        img = RawImage(self.array)
        img.set_pixel(0, 3, math.nan)
        img.set_pixel(5, 6, np.nan)
        img.set_pixel(5, 7, np.nan)

        img.convolve(p)

        # Check that the same pixels are NaN (we ignore those pixels).
        for y in range(self.height):
            for x in range(self.width):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertTrue(math.isnan(img.get_pixel(y, x)))
                else:
                    self.assertFalse(math.isnan(img.get_pixel(y, x)))

    def test_convolve_psf_average(self):
        # Mask out a single pixel.
        img = RawImage(self.array)
        img.mask_pixel(4, 6)

        # Set up a simple "averaging" psf to convolve.
        p = np.zeros((5, 5), dtype=np.single)
        p[1:4, 1:4] = 0.1111111

        img2 = RawImage(img)
        img2.convolve(p)

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

    def test_convolve_psf_orientation(self):
        img = RawImage(self.array.copy())

        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = np.array(psf_data)

        img2 = RawImage(img)
        img2.convolve(p)

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

    # Tests the basic cutout of a stamp from an image.  More advanced stamp
    # construction is done in stamp_creator.cpp and tested in test_search.py.
    def test_make_stamp(self):
        """Tests the basic stamp creation."""
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
        self.assertTrue(image_allclose(stamp.image, stamp2.image, 0.01))

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

        # Check that we throw an error for an empty array or mismatched sizes.
        self.assertRaises(RuntimeError, create_median_image, [])
        img1 = RawImage(np.array([1.0, 2.0, 3.0], dtype=np.single))
        img2 = RawImage(np.array([1.0, 2.0], dtype=np.single))
        self.assertRaises(RuntimeError, create_median_image, [img1, img2])

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

        # Check that we throw an error for an empty array or mismatched sizes.
        self.assertRaises(RuntimeError, create_summed_image, [])
        img1 = RawImage(np.array([1.0, 2.0, 3.0], dtype=np.single))
        img2 = RawImage(np.array([1.0, 2.0], dtype=np.single))
        self.assertRaises(RuntimeError, create_summed_image, [img1, img2])

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

        # Check that we throw an error for an empty array or mismatched sizes.
        self.assertRaises(RuntimeError, create_mean_image, [])
        img1 = RawImage(np.array([1.0, 2.0, 3.0], dtype=np.single))
        img2 = RawImage(np.array([1.0, 2.0], dtype=np.single))
        self.assertRaises(RuntimeError, create_mean_image, [img1, img2])


if __name__ == "__main__":
    unittest.main()
