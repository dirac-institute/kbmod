import logging
import numpy as np
import unittest

from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.search import KB_NO_DATA, Trajectory

from kbmod.image_utils import (
    count_valid_images,
    create_stamps_from_image_stack,
    create_stamps_from_image_stack_xy,
    extract_sci_images_from_stack,
    extract_var_images_from_stack,
    image_allclose,
    image_stack_from_components,
    validate_image_stack,
)


class test_image_utils(unittest.TestCase):
    def test_image_allclose(self):
        """Test that we can compare two images."""
        arr1 = np.arange(12).reshape((3, 4)).astype(np.float32)
        arr2 = np.arange(12).reshape((4, 3)).astype(np.float32)
        self.assertFalse(image_allclose(arr1, arr2))

        arr2 = np.arange(12).reshape((3, 4)).astype(np.float32)
        self.assertTrue(image_allclose(arr1, arr2))

        arr2[0, 1] = arr2[0, 1] + 0.001
        self.assertFalse(image_allclose(arr1, arr2))

        arr1[0, 1] = arr1[0, 1] + 0.001
        self.assertTrue(image_allclose(arr1, arr2))

        arr1[1, 2] = KB_NO_DATA
        self.assertFalse(image_allclose(arr1, arr2))

        arr2[1, 2] = KB_NO_DATA
        self.assertTrue(image_allclose(arr1, arr2))

        arr1[0, 3] = KB_NO_DATA
        arr2[0, 3] = KB_NO_DATA
        self.assertTrue(image_allclose(arr1, arr2))

        arr1[2, 2] = 1.0
        self.assertFalse(image_allclose(arr1, arr2))

    def test_extract_images_from_stack(self):
        """Tests that we can transform an ImageStack into a single numpy array."""
        num_times = 5
        width = 10
        height = 12

        fake_times = np.arange(num_times)
        fake_ds = FakeDataSet(width, height, fake_times, use_seed=True)

        # Check that we can extract the science pixels.
        sci_array = extract_sci_images_from_stack(fake_ds.stack)
        self.assertEqual(sci_array.shape, (num_times, height, width))
        for idx in range(num_times):
            img_data = fake_ds.stack.get_single_image(idx).get_science_array()
            self.assertTrue(np.allclose(sci_array[idx, :, :], img_data))

        # Check that we can extract the variance pixels.
        var_array = extract_var_images_from_stack(fake_ds.stack)
        self.assertEqual(var_array.shape, (num_times, height, width))
        for idx in range(num_times):
            img_data = fake_ds.stack.get_single_image(idx).get_variance_array()
            self.assertTrue(np.allclose(var_array[idx, :, :], img_data))

    def test_image_stack_from_components(self):
        """Tests that we can transform numpy arrays into an ImageStack."""
        num_times = 5
        width = 10
        height = 12

        # Create data as a list of numpy arrays (instead of a 3-d array)
        # to test auto-conversion.
        fake_times = np.arange(num_times)
        fake_sci = [90.0 * np.random.random((height, width)) + 10.0 for _ in range(num_times)]
        fake_var = [0.49 * np.random.random((height, width)) + 0.01 for _ in range(num_times)]
        fake_mask = [np.zeros((height, width)) for _ in range(num_times)]
        fake_psf = [PSF.make_gaussian_kernel(2.0 * (i + 0.1)) for i in range(num_times)]

        im_stack = image_stack_from_components(
            fake_times,
            fake_sci,
            fake_var,
            fake_mask,
            fake_psf,
        )
        self.assertEqual(len(im_stack), num_times)
        self.assertEqual(im_stack.get_height(), height)
        self.assertEqual(im_stack.get_width(), width)
        self.assertEqual(im_stack.get_npixels(), width * height)
        self.assertEqual(im_stack.get_total_pixels(), num_times * width * height)

        for idx in range(num_times):
            img = im_stack.get_single_image(idx)
            self.assertEqual(img.get_width(), width)
            self.assertEqual(img.get_height(), height)
            self.assertAlmostEqual(img.get_obstime(), fake_times[idx])

            # Check that the images are equal. We use a threshold of 0.001 because the
            # RawImage arrays will be converted into single precision floats.
            self.assertTrue(image_allclose(img.get_science_array(), fake_sci[idx], atol=0.001))
            self.assertTrue(image_allclose(img.get_variance_array(), fake_var[idx], atol=0.001))
            self.assertTrue(image_allclose(img.get_mask_array(), fake_mask[idx], atol=0.001))

        # Test that everything still works when we don't pass in a mask or PSFs.
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var)
        self.assertEqual(len(im_stack), num_times)
        self.assertEqual(im_stack.get_height(), height)
        self.assertEqual(im_stack.get_width(), width)

    def test_validate_image_stack(self):
        """Tests that we can validate an ImageStack."""
        # Turn off the warnings for this test.
        logging.disable(logging.CRITICAL)

        # Start with a valid ImageStack.
        num_times = 5
        width = 3
        height = 4
        fake_times = np.arange(num_times)
        fake_sci = [90.0 * np.random.random((height, width)) + 10.0 for _ in range(num_times)]
        fake_var = [0.49 * np.random.random((height, width)) + 0.01 for _ in range(num_times)]
        fake_mask = [np.zeros((height, width)) for _ in range(num_times)]
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var, fake_mask)
        self.assertEqual(im_stack.img_count(), 5)
        self.assertTrue(validate_image_stack(im_stack))

        # Too high flux.
        fake_sci2 = np.full_like(fake_sci, 1.0)
        fake_sci2[1][2][1] = 1e9
        im_stack = image_stack_from_components(fake_times, fake_sci2, fake_var, fake_mask)
        self.assertFalse(validate_image_stack(im_stack))
        self.assertRaises(ValueError, validate_image_stack, im_stack, warn_only=False)

        # Too low flux.
        fake_sci2[1][2][1] = -1e9
        im_stack = image_stack_from_components(fake_times, fake_sci2, fake_var, fake_mask)
        self.assertFalse(validate_image_stack(im_stack))
        self.assertRaises(ValueError, validate_image_stack, im_stack, warn_only=False)

        # Too high variance.
        fake_var2 = np.full_like(fake_var, 1.0)
        fake_var2[1][2][1] = 1e9
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var2, fake_mask)
        self.assertFalse(validate_image_stack(im_stack))
        self.assertRaises(ValueError, validate_image_stack, im_stack, warn_only=False)

        # Too low variance.
        fake_var2[1][2][1] = -1e9
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var2, fake_mask)
        self.assertFalse(validate_image_stack(im_stack))
        self.assertRaises(ValueError, validate_image_stack, im_stack, warn_only=False)

        # Too many masked pixels in an image.
        fake_mask2 = np.full_like(fake_mask, 0)
        fake_mask2[1, :, :] = 1
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var, fake_mask2)
        self.assertFalse(validate_image_stack(im_stack))
        self.assertRaises(ValueError, validate_image_stack, im_stack, warn_only=False)

        # Re-enable warnings.
        logging.disable(logging.NOTSET)

    def test_count_valid_images(self):
        """Tests that we can count the number of valid images in an ImageStack."""
        # Start with a valid ImageStack.
        num_times = 10
        width = 20
        height = 20
        fake_times = np.arange(num_times)
        fake_sci = [90.0 * np.random.random((height, width)) + 10.0 for _ in range(num_times)]
        fake_var = [0.49 * np.random.random((height, width)) + 0.01 for _ in range(num_times)]
        fake_mask = [np.zeros((height, width)) for _ in range(num_times)]
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var, fake_mask)
        self.assertEqual(im_stack.img_count(), 10)
        self.assertTrue(count_valid_images(im_stack), 9)

        # Mask most of the pixels in the science layer 1.
        fake_sci[1][:, 1:width] = np.nan
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var, fake_mask)
        self.assertEqual(im_stack.img_count(), 10)
        self.assertTrue(count_valid_images(im_stack, 0.8), 9)

        # Mask most of the pixels in the mask layers 3 and 7.
        fake_mask[3][:, 1:width] = 1
        fake_mask[7][1:height, :] = 1
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var, fake_mask)
        self.assertEqual(im_stack.img_count(), 10)
        self.assertTrue(count_valid_images(im_stack, 0.8), 7)

        # Mask most of the science pixels in layer 3 (does not change count).
        fake_sci[3][:, 1:width] = np.nan
        im_stack = image_stack_from_components(fake_times, fake_sci, fake_var, fake_mask)
        self.assertEqual(im_stack.img_count(), 10)
        self.assertTrue(count_valid_images(im_stack, 0.8), 7)

    def test_create_stamps_from_image_stack(self):
        # Create a small fake data set for the tests.
        num_times = 10
        fake_times = create_fake_times(num_times, 57130.2, 1, 0.01, 1)
        fake_ds = FakeDataSet(
            25,  # width
            35,  # height
            fake_times,  # time stamps
            1.0,  # noise level
            0.5,  # psf value
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        fake_ds.insert_object(trj)

        # Create stamps from the fake data set and Trajectory.
        stamps = create_stamps_from_image_stack(fake_ds.stack, trj, 1)
        self.assertEqual(len(stamps), num_times)
        for i in range(num_times):
            self.assertEqual(stamps[i].shape, (3, 3))

            # Compare to the (manually computed) trajectory location.
            center_val = fake_ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(center_val):
                self.assertTrue(np.isnan(stamps[i][1, 1]))
            else:
                self.assertAlmostEqual(center_val, stamps[i][1, 1])

        # Check that we can set use_indices to produce only some stamps.
        use_times = [False, True, False, True, True, False, False, False, True, False]
        stamps = create_stamps_from_image_stack(fake_ds.stack, trj, 1, use_times)
        self.assertEqual(len(stamps), np.count_nonzero(use_times))

        stamp_count = 0
        for i in range(num_times):
            if use_times[i]:
                self.assertEqual(stamps[stamp_count].shape, (3, 3))
                center_val = fake_ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
                if np.isnan(center_val):
                    self.assertTrue(np.isnan(stamps[stamp_count][1, 1]))
                else:
                    self.assertAlmostEqual(center_val, stamps[stamp_count][1, 1])
                stamp_count += 1

    def test_create_stamps_from_image_stack_xy(self):
        # Create a small fake data set for the tests.
        num_times = 10
        fake_times = create_fake_times(num_times, 57130.2, 1, 0.01, 1)
        fake_ds = FakeDataSet(
            25,  # width
            35,  # height
            fake_times,  # time stamps
            1.0,  # noise level
            0.5,  # psf value
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        fake_ds.insert_object(trj)

        zeroed_times = np.array(fake_ds.stack.build_zeroed_times())
        xvals = (trj.x + trj.vx * zeroed_times + 0.5).astype(int)
        yvals = (trj.y + trj.vy * zeroed_times + 0.5).astype(int)
        stamps = create_stamps_from_image_stack_xy(fake_ds.stack, 1, xvals, yvals)
        self.assertEqual(len(stamps), num_times)
        for i in range(num_times):
            self.assertEqual(stamps[i].shape, (3, 3))

            pix_val = fake_ds.stack.get_single_image(i).get_science().get_pixel(7 + i, 8 + 2 * i)
            if np.isnan(pix_val):
                self.assertTrue(np.isnan(stamps[i][1, 1]))
            else:
                self.assertAlmostEqual(pix_val, stamps[i][1, 1])

        # Check that we can set use_indices to produce only some stamps.
        use_inds = np.array([1, 2, 3, 5, 6])
        stamps = create_stamps_from_image_stack_xy(fake_ds.stack, 1, xvals, yvals, to_include=use_inds)
        self.assertEqual(len(stamps), len(use_inds))

        for stamp_i, image_i in enumerate(use_inds):
            self.assertEqual(stamps[stamp_i].shape, (3, 3))
            pix_val = (
                fake_ds.stack.get_single_image(image_i)
                .get_science()
                .get_pixel(
                    7 + image_i,
                    8 + 2 * image_i,
                )
            )

            if np.isnan(pix_val):
                self.assertTrue(np.isnan(stamps[stamp_i][1, 1]))
            else:
                self.assertAlmostEqual(pix_val, stamps[stamp_i][1, 1])


if __name__ == "__main__":
    unittest.main()
