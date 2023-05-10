import unittest

from kbmod.masking import (
    BitVectorMasker,
    DictionaryMasker,
    GlobalDictionaryMasker,
    GrowMask,
    ThresholdMask,
    apply_mask_operations,
)
from kbmod.run_search import *
from kbmod.search import *


class test_masking_classes(unittest.TestCase):
    def setUp(self):
        # The configuration parameters.
        self.mask_bits_dict = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }
        self.default_flag_keys = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]

        # Create the a fake layered image.
        self.img_count = 5
        self.dim_x = 20
        self.dim_y = 20
        self.noise_level = 0.1
        self.variance = self.noise_level**2
        self.p = psf(1.0)
        self.imlist = []
        self.time_list = []
        for i in range(self.img_count):
            time = i / self.img_count
            self.time_list.append(time)
            im = layered_image(
                str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i
            )
            self.imlist.append(im)
        self.stack = image_stack(self.imlist)

    def test_threshold_masker(self):
        # Set one science pixel per image above the threshold
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(2 + i, 8, 501.0)
            sci.set_pixel(1 + i, 9, 499.0)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_science(sci)
            self.stack.set_single_image(i, img)

        # With a threshold of 500 one pixel per image should be masked.
        mask = ThresholdMask(500)
        self.stack = mask.apply_mask(self.stack)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if x == 2 + i and y == 8:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_per_image_dictionary_mask(self):
        # Set each mask pixel in a row to one masking reason.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()
            for x in range(self.dim_x):
                msk.set_pixel(x, 3, 2**x)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        # Mask with two keys.
        mask = DictionaryMasker(self.mask_bits_dict, ["BAD", "EDGE"])
        self.stack = mask.apply_mask(self.stack)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 3 and (x == 0 or x == 4):
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

        # Mask with all the default keys.
        mask = DictionaryMasker(self.mask_bits_dict, self.default_flag_keys)
        self.stack = mask.apply_mask(self.stack)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 3 and (x == 0 or x == 4 or x == 7 or x == 8 or x == 15):
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_mask_grow(self):
        # Mask one pixel per image.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()
            for x in range(self.dim_x):
                msk.set_pixel(2 + i, 8, 1)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        # Apply the bit vector based mask and check that one pixel per image is masked.
        self.stack = BitVectorMasker(1, []).apply_mask(self.stack)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    self.assertEqual(sci.pixel_has_data(x, y), x != (2 + i) or y != 8)

        # Grow the mask by two pixels and recheck.
        self.stack = GrowMask(2).apply_mask(self.stack)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    dist = abs(2 + i - x) + abs(y - 8)
                    self.assertEqual(sci.pixel_has_data(x, y), dist > 2)

    def test_global_mask(self):
        # Set each mask pixel in a single row depending on the image number.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()

            # Set key "CR" on every other image.
            if i % 2:
                msk.set_pixel(1, 1, 8)

            # Set key "INTRP" on only one image.
            if i == 0:
                msk.set_pixel(5, 5, 4)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        mask = GlobalDictionaryMasker(self.mask_bits_dict, ["CR", "INTRP"], 2)
        self.stack = mask.apply_mask(self.stack)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    self.assertEqual(sci.pixel_has_data(x, y), x != 1 or y != 1)


class test_run_search_masking(unittest.TestCase):
    def setUp(self):
        # Create the a fake layered image.
        self.img_count = 10
        self.dim_x = 50
        self.dim_y = 50
        self.noise_level = 0.1
        self.variance = self.noise_level**2
        self.p = psf(1.0)
        self.imlist = []
        self.time_list = []
        for i in range(self.img_count):
            time = i / self.img_count
            self.time_list.append(time)
            im = layered_image(
                str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i
            )
            self.imlist.append(im)
        self.stack = image_stack(self.imlist)

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
            # Make the pixel (i, 1) in each image above threshold and (i, 10)
            # right below the threshold.
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(i, 1, 1000.0)
            bad_pixels.append((i, i, 1))
            sci.set_pixel(i, 10, 895.0)

            # Set the "BAD" key on pixel (15 + i, 2) in each image.
            msk = img.get_mask()
            msk.set_pixel(15 + i, 2, 1)
            bad_pixels.append((i, 15 + i, 2))

            # Set the "INTRP" key on pixel (30 + i, 3) in each image.
            # But we won't mark this as a pixel to filter.
            msk.set_pixel(30 + i, 3, 4)

            # Set the "EDGE" key on pixel (2 * i, 4) in every third image.
            if i % 3 == 1:
                msk.set_pixel(2 * i, 4, 16)
                bad_pixels.append((i, 2 * i, 4))

            # Set the "CR" key on pixel (6, 5) in every other image.
            # It will be bad in every image because of global masking.
            if i % 2 == 0:
                msk.set_pixel(6, 5, 8)
            bad_pixels.append((i, 6, 5))

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_science(sci)
            img.set_mask(msk)
            self.stack.set_single_image(i, img)
        bad_set = set(bad_pixels)

        # Do the actual masking.
        rs = run_search(overrides)
        self.stack = rs.do_masking(self.stack)

        # Test the the correct pixels have been masked.
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    # Extend is_bad to check for pixels marked bad in our list OR
                    # pixels adjacent to a bad pixel (from growing the mask 1).
                    is_bad = (
                        (i, x, y) in bad_set
                        or (i, x + 1, y) in bad_set
                        or (i, x - 1, y) in bad_set
                        or (i, x, y + 1) in bad_set
                        or (i, x, y - 1) in bad_set
                    )
                    self.assertEqual(sci.pixel_has_data(x, y), not is_bad)


if __name__ == "__main__":
    unittest.main()
