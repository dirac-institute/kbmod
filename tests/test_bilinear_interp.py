import unittest

import numpy as np

from kbmod.fake_data.fake_data_creator import add_fake_object, make_fake_layered_image
import kbmod.search as kb
from kbmod.trajectory_utils import make_trajectory


class test_bilinear_interp(unittest.TestCase):
    def setUp(self):
        self.im_count = 5
        self.trj = make_trajectory(2, 2, 0.5, 0.5)
        p = kb.PSF(0.05)
        self.images = []
        for c in range(self.im_count):
            im = make_fake_layered_image(10, 10, 0.0, 1.0, c, p)
            add_fake_object(im, self.trj.get_x_pos(c), self.trj.get_y_pos(c), 1, p)
            self.images.append(im)

    def test_pixels(self):
        d = 0.001

        pixels = self.images[0].get_science()

        self.assertAlmostEqual(pixels.get_pixel(2, 2), 1, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(3, 2), 0, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(2, 3), 0, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(1, 2), 0, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(2, 1), 0, delta=d)

        pixels = self.images[1].get_science()
        self.assertAlmostEqual(pixels.get_pixel(2, 2), 0.25, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(3, 2), 0.25, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(2, 3), 0.25, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(3, 3), 0.25, delta=d)
        self.assertAlmostEqual(pixels.get_pixel(2, 1), 0, delta=d)

    def test_pixel_interp(self):
        pixels = np.array([[0.0, 1.2, 0.0], [1.0, 2.0, 1.0]], dtype=np.single)
        im = kb.RawImage(pixels)
        self.assertEqual(im.width, 3)
        self.assertEqual(im.height, 2)
        self.assertEqual(im.npixels, 6)

        # The middle of a pixel should interp to the pixel's value.
        self.assertAlmostEqual(im.interpolate(0.5, 0.5), 0.0, delta=0.001)

        # The point between two pixels should be 50/50.
        self.assertAlmostEqual(im.interpolate(0.5, 1.0), 0.5, delta=0.001)
        self.assertAlmostEqual(im.interpolate(1.0, 0.5), 0.6, delta=0.001)

        # The point between four pixels should be 25/25/25/25
        self.assertAlmostEqual(im.interpolate(1.0, 1.0), 1.05, delta=0.001)

        # Test a part way interpolation.
        self.assertAlmostEqual(im.interpolate(2.5, 0.75), 0.25, delta=0.001)
        self.assertAlmostEqual(im.interpolate(2.5, 1.25), 0.75, delta=0.001)


if __name__ == "__main__":
    unittest.main()
