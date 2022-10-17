import unittest

import numpy
from kbmodpy import kbmod as kb


class test_bilinear_interp(unittest.TestCase):
    def setUp(self):
        self.im_count = 5
        p = kb.psf(0.05)
        self.images = []
        for c in range(self.im_count):
            im = kb.layered_image(str(c), 10, 10, 0.0, 1.0, c, p)
            im.add_object(2 + c * 0.5 + 0.5, 2 + c * 0.5 + 0.5, 1)
            self.images.append(im)

    def test_pixels(self):

        d = 0.001

        pixels = self.images[0].science()
        self.assertAlmostEqual(pixels.item(2, 2), 1, delta=d)
        self.assertAlmostEqual(pixels.item(3, 2), 0, delta=d)
        self.assertAlmostEqual(pixels.item(2, 3), 0, delta=d)
        self.assertAlmostEqual(pixels.item(1, 2), 0, delta=d)
        self.assertAlmostEqual(pixels.item(2, 1), 0, delta=d)

        pixels = self.images[1].science()
        self.assertAlmostEqual(pixels.item(2, 2), 0.25, delta=d)
        self.assertAlmostEqual(pixels.item(3, 2), 0.25, delta=d)
        self.assertAlmostEqual(pixels.item(2, 3), 0.25, delta=d)
        self.assertAlmostEqual(pixels.item(3, 3), 0.25, delta=d)
        self.assertAlmostEqual(pixels.item(2, 1), 0, delta=d)

    def test_pixel_interp(self):
        pixels = numpy.array([[0.0, 1.2, 0.0], [1.0, 2.0, 1.0]])
        im = kb.raw_image(pixels)
        self.assertEqual(im.get_width(), 3)
        self.assertEqual(im.get_height(), 2)
        self.assertEqual(im.get_ppi(), 6)

        # The middle of a pixel should interp to the pixel's value.
        self.assertAlmostEqual(im.get_pixel_interp(0.5, 0.5), 0.0, delta=0.001)

        # The point between two pixels should be 50/50.
        self.assertAlmostEqual(im.get_pixel_interp(0.5, 1.0), 0.5, delta=0.001)
        self.assertAlmostEqual(im.get_pixel_interp(1.0, 0.5), 0.6, delta=0.001)

        # The point between four pixels should be 25/25/25/25
        self.assertAlmostEqual(im.get_pixel_interp(1.0, 1.0), 1.05, delta=0.001)

        # Test a part way interpolation.
        self.assertAlmostEqual(im.get_pixel_interp(2.5, 0.75), 0.25, delta=0.001)
        self.assertAlmostEqual(im.get_pixel_interp(2.5, 1.25), 0.75, delta=0.001)


if __name__ == "__main__":
    unittest.main()
