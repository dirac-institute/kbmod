import unittest

import kbmod.search as kb


class test_regions(unittest.TestCase):
    def setUp(self):
        # test pass thresholds
        p = kb.psf(1.0)
        im = kb.layered_image("", 171, 111, 5.0, 25.0, 0, p)
        stack = kb.image_stack([im])
        self.search = kb.stack_region_search(stack)

        max_img = im.get_science()
        max_img.set_pixel(38, 39, 117)
        max_img.set_pixel(24, 63, 1000)
        max_img.set_pixel(50, 27, 1000)
        max_img.set_pixel(80, 82, 1000)
        self.pooled_max = kb.pooled_image(max_img, kb.pool_max, False)

        min_img = im.get_science()
        self.pooled_min = kb.pooled_image(min_img, kb.pool_min, False)

    def test_extreme_in_region(self):
        self.assertEqual(self.pooled_max.get_pixel(0, 38, 39), 117)
        self.assertLess(self.pooled_min.get_pixel(4, 2, 1), -5.0)
        self.assertEqual(self.search.extreme_in_region(1.6, 1.7, 32, self.pooled_max, kb.pool_max), 117)


if __name__ == "__main__":
    unittest.main()
