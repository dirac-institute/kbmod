from kbmod import *
import numpy as np
import tempfile
import unittest

class test_raw_image(unittest.TestCase):
    def setUp(self):
        self.width = 10
        self.height = 12
        self.img = raw_image(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.img.set_pixel(x, y, float(x+y*self.width))

    def test_create(self):
        self.assertEqual(self.img.get_width(), self.width)
        self.assertEqual(self.img.get_height(), self.height)
        self.assertEqual(self.img.get_ppi(), self.width * self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(self.img.pixel_has_data(x, y))
                self.assertEqual(self.img.get_pixel(x, y),
                                 float(x+y*self.width))

    def test_set_all(self):
        self.img.set_all(15.0)
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(self.img.pixel_has_data(x, y))
                self.assertEqual(self.img.get_pixel(x, y), 15.0)

    def test_make_stamp(self):
        for x in range(self.width):
            for y in range(self.height):
                self.img.set_pixel(x, y, float(x+y*self.width))

        stamp = self.img.create_stamp(2.5, 2.5, 2, True)
        self.assertEqual(stamp.get_height(), 5)
        self.assertEqual(stamp.get_width(), 5)
        for x in range(-2,3):
            for y in range(-2,3):
                self.assertAlmostEqual(stamp.get_pixel(2+x, 2+y),
                                       float((x+2) + (y+2)*self.width),
                                       delta = 0.001)

    def test_extreme_in_region(self):
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 0),
                         float(5 + 5*self.width))
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 1),
                         float(6 + 6*self.width))

        self.assertEqual(self.img.extreme_in_region(1, 0, 3, 4, 0),
                         1.0)
        self.assertEqual(self.img.extreme_in_region(1, 0, 3, 4, 1),
                         float(3 + 4*self.width))

        self.img.set_pixel(5, 5, KB_NO_DATA)
        self.img.set_pixel(5, 6, KB_NO_DATA)
        self.img.set_pixel(6, 6, KB_NO_DATA)
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 0),
                         float(6 + 5*self.width))
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 1),
                         float(6 + 5*self.width))

        self.img.set_pixel(6, 5, KB_NO_DATA)
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 0),
                         KB_NO_DATA)
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 1),
                         KB_NO_DATA)

    def test_create_median_image(self):
        img1 = raw_image(np.array([[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]]))
        img2 = raw_image(np.array([[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]]))
        img3 = raw_image(np.array([[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]]))
        vect = [img1, img2, img3]
        median_image = create_median_image(vect)

        self.assertEqual(median_image.get_width(), 2)
        self.assertEqual(median_image.get_height(), 3)
        self.assertAlmostEqual(median_image.get_pixel(0, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(1, 0), -1.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(0, 1), 2.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(1, 1), 3.5, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(0, 2), 4.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(1, 2), 3.1, delta=1e-6)

    def test_create_summed_image(self):
        img1 = raw_image(np.array([[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]]))
        img2 = raw_image(np.array([[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]]))
        img3 = raw_image(np.array([[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]]))
        vect = [img1, img2, img3]
        summed_image = create_summed_image(vect)

        self.assertEqual(summed_image.get_width(), 2)
        self.assertEqual(summed_image.get_height(), 3)
        self.assertAlmostEqual(summed_image.get_pixel(0, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(summed_image.get_pixel(1, 0), -3.0, delta=1e-6)
        self.assertAlmostEqual(summed_image.get_pixel(0, 1), 6.0, delta=1e-6)
        self.assertAlmostEqual(summed_image.get_pixel(1, 1), 9.5, delta=1e-6)
        self.assertAlmostEqual(summed_image.get_pixel(0, 2), 8.8, delta=1e-6)
        self.assertAlmostEqual(summed_image.get_pixel(1, 2), 9.4, delta=1e-6)
        
if __name__ == '__main__':
   unittest.main()

