import math
import tempfile
import unittest

from kbmod import *


def extreme_of_four(image, x, y, compute_max=True):
    """
    A helper function that computes the maximum or minumum
    of a 2x2 square of pixels starting at (x, y).

    Arguments:
      image - a raw_image object holding the pixels
      x - the x coordinate
      y - the y coordinate
      compute_max - Boolean indicating whether to
                    compute max ot min.

    Returns:
      The maxmium or minimum pixel value of the 2x2 square
      (excluding pixels with no data). Returns None if all
      the pixels do not have values.
    """
    values = []
    if image.pixel_has_data(x, y):
        values.append(image.get_pixel(x, y))
    if image.pixel_has_data(x + 1, y):
        values.append(image.get_pixel(x + 1, y))
    if image.pixel_has_data(x, y + 1):
        values.append(image.get_pixel(x, y + 1))
    if image.pixel_has_data(x + 1, y + 1):
        values.append(image.get_pixel(x + 1, y + 1))

    if len(values) == 0:
        return None
    if compute_max:
        return max(values)
    else:
        return min(values)

def extreme_of_nine(image, x, y, compute_max=True):
    """
    A helper function that computes the maximum or minumum
    of a 3x3 square of pixels centered at (x, y).

    Arguments:
      image - a raw_image object holding the pixels
      x - the x coordinate
      y - the y coordinate
      compute_max - Boolean indicating whether to 
                    compute max ot min.

    Returns:
      The maxmium or minimum pixel value of the 3x3 square
      (excluding pixels with no data). Returns None if all
      the pixels do not have values.
    """
    values = []
    for xd in range(-1, 2):
        for yd in range(-1, 2):
            if image.pixel_has_data(x + xd, y + yd):
                values.append(image.get_pixel(x + xd, y + yd))

    if len(values) == 0:
        return None
    if compute_max:
        return max(values)
    else:
        return min(values)
    
class test_pooled_image(unittest.TestCase):
    def setUp(self):
        self.width = 16
        self.height = 16
        self.base_image = raw_image(self.width, self.height)

        # Set most of the pixels in ascending order.
        for x in range(self.width):
            for y in range(self.height):
                self.base_image.set_pixel(x, y, (float)(x + y * self.width))

        # Set a few pixels to no data.
        self.base_image.set_pixel(2, 2, -9999.0)
        self.base_image.set_pixel(2, 3, -9999.0)
        self.base_image.set_pixel(2, 4, -9999.0)
        self.base_image.set_pixel(3, 2, -9999.0)
        self.base_image.set_pixel(3, 3, -9999.0)
        self.base_image.set_pixel(3, 4, -9999.0)

    def test_create_max(self):
        pooled = pooled_image(self.base_image, pool_max, False)
        self.assertEqual(pooled.get_base_width(), self.width)
        self.assertEqual(pooled.get_base_height(), self.height)
        self.assertEqual(pooled.get_base_ppi(), self.width * self.height)
        self.assertEqual(pooled.num_levels(), 5)

        level_w = self.width
        level_h = self.height
        last = None
        for lvl in range(pooled.num_levels()):
            img = pooled.get_image(lvl)
            self.assertEqual(img.get_width(), level_w)
            self.assertEqual(img.get_height(), level_h)

            if last is not None:
                for x in range(level_w):
                    for y in range(level_h):
                        value = extreme_of_four(last, 2 * x, 2 * y, True)
                        if value is None:
                            self.assertFalse(img.pixel_has_data(x, y))
                        else:
                            self.assertEqual(img.get_pixel(x, y), value)

            level_w = (int)(level_w / 2)
            level_h = (int)(level_h / 2)
            last = img

    def test_create_min(self):
        pooled = pooled_image(self.base_image, pool_min, False)
        self.assertEqual(pooled.get_base_width(), self.width)
        self.assertEqual(pooled.get_base_height(), self.height)
        self.assertEqual(pooled.get_base_ppi(), self.width * self.height)
        self.assertEqual(pooled.num_levels(), 5)

        level_w = self.width
        level_h = self.height
        last = None
        for lvl in range(pooled.num_levels()):
            img = pooled.get_image(lvl)
            self.assertEqual(img.get_width(), level_w)
            self.assertEqual(img.get_height(), level_h)

            if last is not None:
                for x in range(level_w):
                    for y in range(level_h):
                        value = extreme_of_four(last, 2 * x, 2 * y, False)
                        if value is None:
                            self.assertFalse(img.pixel_has_data(x, y))
                        else:
                            self.assertEqual(img.get_pixel(x, y), value)

            level_w = (int)(level_w / 2)
            level_h = (int)(level_h / 2)
            last = img

    def test_create_symmetric_max(self):
        pooled = pooled_image(self.base_image, pool_max, True)
        self.assertEqual(pooled.get_base_width(), self.width)
        self.assertEqual(pooled.get_base_height(), self.height)
        self.assertEqual(pooled.get_base_ppi(), self.width * self.height)
        self.assertEqual(pooled.num_levels(), 5)

        level_w = self.width
        level_h = self.height
        last = None
        for lvl in range(pooled.num_levels()):
            img = pooled.get_image(lvl)
            self.assertEqual(img.get_width(), level_w)
            self.assertEqual(img.get_height(), level_h)

            if last is not None:
                for x in range(level_w):
                    for y in range(level_h):
                        value = extreme_of_nine(last, 2*x, 2*y, True)
                        if value is None:
                            self.assertFalse(img.pixel_has_data(x, y))
                        else:
                            self.assertEqual(img.get_pixel(x, y), value)

            level_w = (int)(level_w / 2)
            level_h = (int)(level_h / 2)
            last = img

    def test_create_symmetric_min(self):
        pooled = pooled_image(self.base_image, pool_min, True)
        self.assertEqual(pooled.get_base_width(), self.width)
        self.assertEqual(pooled.get_base_height(), self.height)
        self.assertEqual(pooled.get_base_ppi(), self.width * self.height)
        self.assertEqual(pooled.num_levels(), 5)

        level_w = self.width
        level_h = self.height
        last = None
        for lvl in range(pooled.num_levels()):
            img = pooled.get_image(lvl)
            self.assertEqual(img.get_width(), level_w)
            self.assertEqual(img.get_height(), level_h)

            if last is not None:
                for x in range(level_w):
                    for y in range(level_h):
                        value = extreme_of_nine(last, 2*x, 2*y, False)
                        if value is None:
                            self.assertFalse(img.pixel_has_data(x, y))
                        else:
                            self.assertEqual(img.get_pixel(x, y), value)

            level_w = (int)(level_w / 2)
            level_h = (int)(level_h / 2)
            last = img

    def test_create_odd_dim(self):
        base_image = raw_image(11, 13)
        base_image.set_all(1.0)
        pooled = pooled_image(base_image, pool_max, False)

        self.assertEqual(pooled.get_base_width(), 11)
        self.assertEqual(pooled.get_base_height(), 13)
        self.assertEqual(pooled.get_base_ppi(), 11 * 13)
        self.assertEqual(pooled.num_levels(), 5)

    def test_repool_area(self):
        pooled = pooled_image(self.base_image, pool_max, False)
        first_level_img = pooled.get_image(0)

        # Mask a few new points.
        first_level_img.set_pixel(12, 12, -9999.0)
        first_level_img.set_pixel(12, 13, -9999.0)
        first_level_img.set_pixel(13, 12, -9999.0)
        first_level_img.set_pixel(13, 13, -9999.0)

        # Try both repooling and creating a new pooled image.
        pooled2 = pooled_image(first_level_img, pool_max, False)
        pooled.repool_area(12, 12, 2)
        self.assertEqual(pooled.num_levels(), pooled2.num_levels())

        for lvl in range(pooled.num_levels()):
            img1 = pooled.get_image(lvl)
            img2 = pooled.get_image(lvl)
            for x in range(img1.get_width()):
                for y in range(img1.get_height()):
                    self.assertEqual(img1.get_pixel(x, y), img2.get_pixel(x, y))

    def test_mapped_pixel_at_depth(self):
        pooled = pooled_image(self.base_image, pool_min, False)

        # The pixel 5, 6 has value 5 + 6 * width at level 0
        self.assertEqual(pooled.get_mapped_pixel_at_depth(0, 5, 6), float(5 + 6 * self.width))

        # It maps to pixel (2, 3) at level 1 with value = 4 + 6 * width
        self.assertEqual(pooled.get_mapped_pixel_at_depth(1, 5, 6), float(4 + 6 * self.width))

        # It maps to pixel (1, 3) at level 2 with value = 4 + 4 * width
        self.assertEqual(pooled.get_mapped_pixel_at_depth(2, 5, 6), float(4 + 4 * self.width))

    def test_contains_pixel(self):
        pooled = pooled_image(self.base_image, pool_min, False)

        # Run basic tests at depth=0
        self.assertTrue(pooled.contains_pixel(0, 1, 1, 1, 1))
        self.assertFalse(pooled.contains_pixel(0, 1, 1, 1, 2))
        self.assertFalse(pooled.contains_pixel(0, 1, 1, 2, 1))
        self.assertFalse(pooled.contains_pixel(0, 2, 2, 1, 1))

        # Pixel (5, 6) maps to (2, 3) at depth=1 and (1, 1) at depth=2.
        self.assertTrue(pooled.contains_pixel(1, 2, 3, 5, 6))
        self.assertFalse(pooled.contains_pixel(1, 1, 3, 5, 6))
        self.assertFalse(pooled.contains_pixel(1, 2, 4, 5, 6))
        self.assertTrue(pooled.contains_pixel(2, 1, 1, 5, 6))
        self.assertFalse(pooled.contains_pixel(2, 1, 2, 5, 6))
        self.assertFalse(pooled.contains_pixel(2, 0, 1, 5, 6))

        # Pixel (7, 1) maps to (3, 0) at depth=1, (1, 0) at depth=2,
        # and (0, 0) at depth=3.
        self.assertTrue(pooled.contains_pixel(1, 3, 0, 7, 1))
        self.assertFalse(pooled.contains_pixel(1, 1, 1, 7, 1))
        self.assertFalse(pooled.contains_pixel(1, 2, 4, 7, 1))
        self.assertTrue(pooled.contains_pixel(2, 1, 0, 7, 1))
        self.assertFalse(pooled.contains_pixel(2, 1, 1, 7, 1))
        self.assertFalse(pooled.contains_pixel(2, 0, 1, 7, 1))
        self.assertTrue(pooled.contains_pixel(3, 0, 0, 7, 1))
        self.assertFalse(pooled.contains_pixel(3, 1, 1, 7, 1))
        self.assertFalse(pooled.contains_pixel(3, 1, 0, 7, 1))

    def test_pool_multiple(self):
        to_pool = []
        for i in range(5):
            img = raw_image(16, 16)
            img.set_all(float(i))
            to_pool.append(img)

        destination = pool_multiple_images(to_pool, pool_max, False)

        self.assertEqual(len(destination), 5)
        for i in range(5):
            pi = destination[i]
            self.assertEqual(pi.num_levels(), 5)
            self.assertEqual(pi.get_pixel(0, 0, 0), float(i))

    def test_pixel_distance(self):
        pooled = pooled_image(self.base_image, pool_max, False)

        # (0, 0) to (1, 1) at depth=0
        res = pooled.get_pixel_dist_bounds(0, 0, 0, 1, 1)
        self.assertAlmostEqual(res[0], 0.0, delta=1e-5)
        self.assertAlmostEqual(res[1], math.sqrt(8.0), delta=1e-5)

        # (1, 2) to (3, 6) at depth=0
        res = pooled.get_pixel_dist_bounds(0, 1, 2, 3, 6)
        self.assertAlmostEqual(res[0], math.sqrt(10.0), delta=1e-5)
        self.assertAlmostEqual(res[1], math.sqrt(34.0), delta=1e-5)

        # (1, 2) to (3, 6) at depth=1
        res = pooled.get_pixel_dist_bounds(1, 1, 2, 3, 6)
        self.assertAlmostEqual(res[0], math.sqrt(40.0), delta=1e-5)
        self.assertAlmostEqual(res[1], math.sqrt(136.0), delta=1e-5)

        # (1, 2) to (3, 6) at depth=2
        res = pooled.get_pixel_dist_bounds(2, 1, 2, 3, 6)
        self.assertAlmostEqual(res[0], math.sqrt(160.0), delta=1e-5)
        self.assertAlmostEqual(res[1], math.sqrt(544.0), delta=1e-5)

        # (3, 6) to (1, 2) at depth=2
        res = pooled.get_pixel_dist_bounds(2, 3, 6, 1, 2)
        self.assertAlmostEqual(res[0], math.sqrt(160.0), delta=1e-5)
        self.assertAlmostEqual(res[1], math.sqrt(544.0), delta=1e-5)

        # (7, 5) to (7, 5) at depth=2
        res = pooled.get_pixel_dist_bounds(2, 7, 5, 7, 5)
        self.assertAlmostEqual(res[0], math.sqrt(0.0), delta=1e-5)
        self.assertAlmostEqual(res[1], math.sqrt(32.0), delta=1e-5)


if __name__ == "__main__":
    unittest.main()
