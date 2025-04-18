import numpy as np
import unittest

from kbmod.core.stamp_utils import (
    coadd_mean,
    coadd_median,
    coadd_sum,
    coadd_weighted,
    extract_curve_values,
    extract_stamp,
    extract_stamp_stack,
)


class test_stamp_utils(unittest.TestCase):
    def test_extract_single_stamp(self):
        """Tests the basic stamp creation."""
        width = 12
        height = 10
        sci_data = np.arange(0, width * height, dtype=np.single).reshape(1, height, width)

        # Test a stamp at the corner of the image. But entirely within the image.
        stamp = extract_stamp(sci_data[0], 2, 2, 2)
        self.assertEqual(stamp.shape, (5, 5))
        self.assertTrue(np.allclose(stamp, sci_data[0, 0:5, 0:5], equal_nan=True))

        # Test a stamp that is not at the corner.
        stamp2 = extract_stamp(sci_data[0], 8, 5, 1)
        self.assertEqual(stamp2.shape, (3, 3))
        self.assertTrue(np.allclose(stamp2, sci_data[0, 4:7, 7:10], equal_nan=True))

        # Test a stamp that goes out of bounds.
        stamp3 = extract_stamp(sci_data[0], 11, 0, 1)
        expected = np.array([[np.nan, np.nan, np.nan], [10.0, 11.0, np.nan], [22.0, 23.0, np.nan]])
        self.assertTrue(np.allclose(stamp3, expected, equal_nan=True))

        # Test a stamp that is completely out of bounds.
        stamp4 = extract_stamp(sci_data[0], 20, 20, 1)
        expected = np.full((3, 3), np.nan)
        self.assertTrue(np.allclose(stamp4, expected, equal_nan=True))

        # Test a stamp that is completely out of bounds along a second direction.
        stamp5 = extract_stamp(sci_data[0], -5, -5, 1)
        expected = np.full((3, 3), np.nan)
        self.assertTrue(np.allclose(stamp5, expected, equal_nan=True))

        # Test a stamp that overlaps at a single corner pixel.
        stamp6 = extract_stamp(sci_data[0], -1, -1, 1)
        expected = np.full((3, 3), np.nan)
        expected[2][2] = 0.0
        self.assertTrue(np.allclose(stamp6, expected, equal_nan=True))

    def test_extract_stamp_stack(self):
        """Tests the basic stamp creation for a stack of images."""
        num_times = 4
        width = 12
        height = 10
        times = np.arange(num_times)
        data = np.arange(0, num_times * width * height).reshape(num_times, height, width)

        # Test that we can extract a stack of stamps.
        x_vals = (-2.0 + 2.0 * times + 0.5).astype(int)
        y_vals = np.full(num_times, 1.0 + 0.5).astype(int)
        stamp_array = extract_stamp_stack(data, x_vals, y_vals, 2)

        self.assertEqual(stamp_array.shape, (num_times, 5, 5))
        center_vals = stamp_array[:, 2, 2]
        expected = np.array([np.nan, 132.0, 254.0, 376.0])

        self.assertTrue(np.allclose(center_vals, expected, equal_nan=True))

        # Test that we fail with a bad radius.
        self.assertRaises(ValueError, extract_stamp_stack, data, x_vals, y_vals, -1)

        # Test that we fail with the wrong number of x_vals or y_vals.
        self.assertRaises(ValueError, extract_stamp_stack, data, x_vals[:-1], y_vals, 2)
        self.assertRaises(ValueError, extract_stamp_stack, data, x_vals, y_vals[1:], 2)

    def test_extract_stamp_stack__list(self):
        """Tests the basic stamp creation for a stack of images as a list."""
        num_times = 4
        width = 12
        height = 10
        times = np.arange(num_times)
        data = np.arange(0, num_times * width * height).reshape(num_times, height, width)
        data_list = [data[i] for i in range(num_times)]

        # Test that we can extract a stack of stamps.
        x_vals = (-2.0 + 2.0 * times + 0.5).astype(int)
        y_vals = np.full(num_times, 1.0 + 0.5).astype(int)
        stamp_array = extract_stamp_stack(data_list, x_vals, y_vals, 2)

        self.assertTrue(isinstance(stamp_array, list))
        self.assertEqual(len(stamp_array), num_times)
        for t in range(num_times):
            self.assertEqual(stamp_array[t].shape, (5, 5))

        # Check the center values.
        self.assertTrue(np.isnan(stamp_array[0][2, 2]))
        self.assertAlmostEqual(stamp_array[1][2, 2], 132.0)
        self.assertAlmostEqual(stamp_array[2][2, 2], 254.0)
        self.assertAlmostEqual(stamp_array[3][2, 2], 376.0)

    def test_make_coadds_simple(self):
        times = np.array([0.0, 1.0, 2.0])
        psf = np.array([1.0])

        # Create an image set with three 3x3 images.
        sci1 = np.array([[0, np.nan, np.nan], [0, np.nan, 0.5], [0, 1, 0.5]]).astype(np.float32)
        sci2 = np.array([[1, np.nan, 0.5], [1, 2, 0.5], [1, 2, 0.5]]).astype(np.float32)
        sci3 = np.array([[2, 3, 0.5], [2, 3, 0.5], [2, 3, 0.5]]).astype(np.float32)
        sci = np.array([sci1, sci2, sci3])

        var1 = np.full((3, 3), 0.1).astype(np.float32)
        var2 = np.full((3, 3), 0.2).astype(np.float32)
        var3 = np.full((3, 3), 0.5).astype(np.float32)
        var = np.array([var1, var2, var3])

        # One trajectory right in the image's middle.
        x_vals = np.array([1.0, 1.0, 1.0])
        y_vals = np.array([1.0, 1.0, 1.0])
        stamp_stack = extract_stamp_stack(sci, x_vals, y_vals, 1)

        # Compute and check the coadds
        sum_coadd = coadd_sum(stamp_stack)
        expected_sum = np.array([[3.0, 3.0, 1.0], [3.0, 5.0, 1.5], [3.0, 6.0, 1.5]]).astype(np.float32)
        self.assertTrue(np.allclose(sum_coadd, expected_sum, atol=1e-5))

        mean_coadd = coadd_mean(stamp_stack)
        expected_mean = np.array([[1.0, 3.0, 0.5], [1.0, 2.5, 0.5], [1.0, 2.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(mean_coadd, expected_mean, atol=1e-5))

        median_coadd = coadd_median(stamp_stack)
        # Note torch's nanmedian implementation is different from numpy's in that
        # it returns the first value in the case of an even number of elements.
        expected_median = np.array([[1.0, 3.0, 0.5], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(median_coadd, expected_median, atol=1e-5))

        var_stack = extract_stamp_stack(var, x_vals, y_vals, 1)
        weighted_coadd = coadd_weighted(stamp_stack, var_stack)
        expected_weighted = np.array(
            [
                [0.5294117647058824, 3.0, 0.5],
                [0.5294117647058824, 2.2857142857142856, 0.5],
                [0.5294117647058824, 1.5294117647058822, 0.5],
            ]
        )
        self.assertTrue(np.allclose(weighted_coadd, expected_weighted, atol=1e-5))

        # Compute and check the coadds when we mask out the third image.
        # Note that there are NO valid values of pixel (0, 2), so we use 0.0.
        mask = np.array([True, True, False])
        stamp_stack = extract_stamp_stack(sci, x_vals, y_vals, 1, time_mask=mask)
        var_stack = extract_stamp_stack(var, x_vals, y_vals, 1, time_mask=mask)

        sum_coadd = coadd_sum(stamp_stack)
        expected_sum = np.array([[1.0, 0.0, 0.5], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0]]).astype(np.float32)
        self.assertTrue(np.allclose(sum_coadd, expected_sum, atol=1e-5))

        mean_coadd = coadd_mean(stamp_stack)
        expected_mean = np.array([[0.5, 0.0, 0.5], [0.5, 2.0, 0.5], [0.5, 1.5, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(mean_coadd, expected_mean, atol=1e-5))

        median_coadd = coadd_median(stamp_stack)
        # Note torch's nanmedian implementation is different from numpy's in that
        # it returns the first value in the case of an even number of elements.
        expected_median = np.array([[0.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 1.0, 0.5]]).astype(np.float32)
        self.assertTrue(np.allclose(median_coadd, expected_median, atol=1e-5))

        weighted_coadd = coadd_weighted(stamp_stack, var_stack)
        expected_weighted = np.array(
            [
                [0.3333333333333333, 0.0, 0.5],
                [0.3333333333333333, 2.0, 0.5],
                [0.3333333333333333, 1.3333333333333333, 0.5],
            ]
        )
        self.assertTrue(np.allclose(weighted_coadd, expected_weighted, atol=1e-5))

    def test_extract_curve_values(self):
        # Create an image set with three 3x3 images.
        sci1 = np.array([[0, np.nan, np.nan], [0, np.nan, 0.5], [0, 1, 0.5]]).astype(np.float32)
        sci2 = np.array([[1, np.nan, 0.5], [1, 2, 0.5], [1, 2, 0.5]]).astype(np.float32)
        sci3 = np.array([[2, 3, 0.5], [2, 3, 0.5], [2, 3, 0.5]]).astype(np.float32)
        sci = np.array([sci1, sci2, sci3])

        # One trajectory right in the image's middle.
        x_vals = np.array([1.0, 1.0, 1.0])
        y_vals = np.array([1.0, 1.0, 1.0])

        # Compute and check the curve values.
        curve_values = extract_curve_values(sci, x_vals, y_vals)
        self.assertTrue(np.isnan(curve_values[0]))
        self.assertAlmostEqual(curve_values[1], 2.0)
        self.assertAlmostEqual(curve_values[2], 3.0)

        # Test that we can handle a curve that goes out of bounds.
        x_vals = np.array([0, 0, 5])
        y_vals = np.array([0, 0, 0])
        curve_values = extract_curve_values(sci, x_vals, y_vals)
        self.assertAlmostEqual(curve_values[0], 0.0)
        self.assertAlmostEqual(curve_values[1], 1.0)
        self.assertTrue(np.isnan(curve_values[2]))

        # Test another curve that goes out of bounds.
        x_vals = np.array([0, 0, 0])
        y_vals = np.array([-2, 0, 0])
        curve_values = extract_curve_values(sci, x_vals, y_vals)
        self.assertTrue(np.isnan(curve_values[0]))
        self.assertAlmostEqual(curve_values[1], 1.0)
        self.assertAlmostEqual(curve_values[2], 2.0)


if __name__ == "__main__":
    unittest.main()
