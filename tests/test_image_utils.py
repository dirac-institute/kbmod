from itertools import product

import logging
import numpy as np
import unittest

from kbmod.core.psf import PSF
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.search import (
    ImageStack,
    LayeredImage,
    Trajectory,
    KB_NO_DATA,
)
from kbmod.image_utils import (
    extract_sci_images_from_stack,
    extract_var_images_from_stack,
    image_allclose,
    image_stack_from_components,
    stat_image_stack,
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
            img_data = fake_ds.stack.get_single_image(idx).get_science().image
            self.assertTrue(np.allclose(sci_array[idx, :, :], img_data))

        # Check that we can extract the variance pixels.
        var_array = extract_var_images_from_stack(fake_ds.stack)
        self.assertEqual(var_array.shape, (num_times, height, width))
        for idx in range(num_times):
            img_data = fake_ds.stack.get_single_image(idx).get_variance().image
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
            self.assertTrue(image_allclose(img.get_science().image, fake_sci[idx], atol=0.001))
            self.assertTrue(image_allclose(img.get_variance().image, fake_var[idx], atol=0.001))
            self.assertTrue(image_allclose(img.get_mask().image, fake_mask[idx], atol=0.001))

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


if __name__ == "__main__":
    unittest.main()
