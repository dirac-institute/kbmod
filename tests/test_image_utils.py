from itertools import product
import numpy as np
import unittest

from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.search import (
    ImageStack,
    LayeredImage,
    PSF,
    StampType,
    Trajectory,
    KB_NO_DATA,
)
from kbmod.image_utils import (
    extract_sci_images_from_stack,
    extract_var_images_from_stack,
)


class test_image_utils(unittest.TestCase):
    # Tests the basic cutout of a stamp from an numpy array of pixel data.
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


if __name__ == "__main__":
    unittest.main()
