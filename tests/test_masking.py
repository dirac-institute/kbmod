import unittest

from kbmod.configuration import SearchConfiguration
from kbmod.masking import apply_mask_operations
from kbmod.search import *


class test_run_search_masking(unittest.TestCase):
    def setUp(self):
        # Create the a fake layered image.
        self.img_count = 10
        self.dim_x = 50
        self.dim_y = 50
        self.noise_level = 0.1
        self.variance = self.noise_level**2
        self.p = PSF(1.0)
        self.imlist = []
        self.time_list = []
        for i in range(self.img_count):
            time = i / self.img_count
            self.time_list.append(time)
            im = LayeredImage(self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i)
            self.imlist.append(im)
        self.stack = ImageStack(self.imlist)

    def test_apply_masks(self):
        overrides = {
            "im_filepath": "./",
            "flag_keys": ["BAD", "EDGE", "SUSPECT"],
            "repeated_flag_keys": ["CR"],
            "mask_grow": 1,
            "mask_threshold": 900.0,
        }

        bad_pixels = []
        for i in range(self.img_count):
            # Make the pixel (1, i) in each image above threshold and (10, i)
            # right below the threshold.
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(1, i, 1000.0)
            bad_pixels.append((i, 1, i))
            sci.set_pixel(10, i, 895.0)

            # Set the "BAD" key on pixel (2, 15 + i) in each image.
            msk = img.get_mask()
            msk.set_pixel(2, 15 + i, 1)
            bad_pixels.append((i, 2, 15 + i))

            # Set the "INTRP" key on pixel (3, 30 + i) in each image.
            # But we won't mark this as a pixel to filter.
            msk.set_pixel(3, 30 + i, 4)

            # Set the "EDGE" key on pixel (2 * i, 4) in every third image.
            if i % 3 == 1:
                msk.set_pixel(4, 2 * i, 16)
                bad_pixels.append((i, 4, 2 * i))

            # Set the "CR" key on pixel (5, 6) in every other image.
            # It will be bad in every image because of global masking.
            if i % 2 == 0:
                msk.set_pixel(5, 6, 8)
            bad_pixels.append((i, 5, 6))

        bad_set = set(bad_pixels)

        config = SearchConfiguration()
        config.set_multiple(overrides)

        # Do the actual masking.
        self.stack = apply_mask_operations(config, self.stack)

        # Test the the correct pixels have been masked.
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for y in range(self.dim_y):
                for x in range(self.dim_x):
                    # Extend is_bad to check for pixels marked bad in our list OR
                    # pixels adjacent to a bad pixel (from growing the mask 1).
                    is_bad = (
                        (i, y, x) in bad_set
                        or (i, y + 1, x) in bad_set
                        or (i, y - 1, x) in bad_set
                        or (i, y, x + 1) in bad_set
                        or (i, y, x - 1) in bad_set
                    )
                    self.assertEqual(sci.pixel_has_data(y, x), not is_bad)


if __name__ == "__main__":
    unittest.main()
