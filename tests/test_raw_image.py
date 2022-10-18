import tempfile
import unittest

import numpy as np

from kbmod import *


class test_raw_image(unittest.TestCase):
    def setUp(self):
        self.width = 10
        self.height = 12
        self.img = raw_image(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.img.set_pixel(x, y, float(x + y * self.width))

    def test_create(self):
        self.assertEqual(self.img.get_width(), self.width)
        self.assertEqual(self.img.get_height(), self.height)
        self.assertEqual(self.img.get_ppi(), self.width * self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(self.img.pixel_has_data(x, y))
                self.assertEqual(self.img.get_pixel(x, y), float(x + y * self.width))

    def test_set_all(self):
        self.img.set_all(15.0)
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(self.img.pixel_has_data(x, y))
                self.assertEqual(self.img.get_pixel(x, y), 15.0)

    def test_get_bounds(self):
        self.img.set_all(10.0)
        self.img.set_pixel(5, 6, 0.1)
        self.img.set_pixel(5, 7, KB_NO_DATA)
        self.img.set_pixel(3, 1, 100.0)
        self.img.set_pixel(4, 4, KB_NO_DATA)
        self.img.set_pixel(5, 5, KB_NO_DATA)

        bnds = self.img.compute_bounds()
        self.assertAlmostEqual(bnds[0], 0.1, delta=1e-6)
        self.assertAlmostEqual(bnds[1], 100.0, delta=1e-6)

    def test_convolve_psf_identity(self):
        psf_data = [[0.0 for _ in range(3)] for _ in range(3)]
        psf_data[1][1] = 1.0
        p = psf(np.array(psf_data))

        # Convolve the identity PSF.
        self.img.convolve(p)

        # Check that the image is unchanged.
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(self.img.pixel_has_data(x, y))
                self.assertEqual(self.img.get_pixel(x, y), float(x + y * self.width))

    def test_convolve_psf_mask(self):
        p = psf(1.0)

        # Mask out three pixels.
        self.img.set_pixel(5, 6, KB_NO_DATA)
        self.img.set_pixel(0, 3, KB_NO_DATA)
        self.img.set_pixel(5, 7, KB_NO_DATA)

        self.img.convolve(p)

        # Check that the image is unchanged.
        for x in range(self.width):
            for y in range(self.height):
                if (x == 5 and y == 6) or (x == 0 and y == 3) or (x == 5 and y == 7):
                    self.assertFalse(self.img.pixel_has_data(x, y))
                else:
                    self.assertTrue(self.img.pixel_has_data(x, y))

    def test_convolve_psf_average(self):
        # Set up a simple "averaging" psf to convolve.
        psf_data = [[0.0 for _ in range(5)] for _ in range(5)]
        for x in range(1, 4):
            for y in range(1, 4):
                psf_data[x][y] = 0.1111111
        p = psf(np.array(psf_data))
        self.assertAlmostEqual(p.get_sum(), 1.0, delta=0.00001)

        # Make a clean version of the image for the average function.
        img2 = raw_image(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                img2.set_pixel(x, y, self.img.get_pixel(x, y))

        # Convolve the psf with the copy of the image.
        img2.convolve(p)

        for x in range(self.width):
            for y in range(self.height):
                # Compute the weighted average around (x, y)
                # in the original image.
                running_sum = 0.0
                count = 0.0
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        value = self.img.get_pixel(x + i, y + j)
                        psf_value = 0.1111111
                        if i == -2 or i == 2 or j == -2 or j == 2:
                            psf_value = 0.0

                        if value != KB_NO_DATA:
                            running_sum += psf_value * value
                            count += psf_value
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                self.assertAlmostEqual(img2.get_pixel(x, y), ave, delta=0.001)

    def test_convolve_psf_orientation(self):
        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = psf(np.array(psf_data))

        # Make a clean version of the image for the average function.
        img2 = raw_image(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                img2.set_pixel(x, y, self.img.get_pixel(x, y))

        # Convolve the psf with the copy of the image.
        img2.convolve(p)

        for x in range(self.width):
            for y in range(self.height):
                running_sum = 0.5 * self.img.get_pixel(x, y)
                count = 0.5
                if self.img.pixel_has_data(x + 1, y):
                    running_sum += 0.4 * self.img.get_pixel(x + 1, y)
                    count += 0.4
                if self.img.pixel_has_data(x, y + 1):
                    running_sum += 0.1 * self.img.get_pixel(x, y + 1)
                    count += 0.1
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                self.assertAlmostEqual(img2.get_pixel(x, y), ave, delta=0.001)

    def test_make_stamp(self):
        for x in range(self.width):
            for y in range(self.height):
                self.img.set_pixel(x, y, float(x + y * self.width))

        stamp = self.img.create_stamp(2.5, 2.5, 2, True, False)
        self.assertEqual(stamp.get_height(), 5)
        self.assertEqual(stamp.get_width(), 5)
        for x in range(-2, 3):
            for y in range(-2, 3):
                self.assertAlmostEqual(
                    stamp.get_pixel(2 + x, 2 + y), float((x + 2) + (y + 2) * self.width), delta=0.001
                )

    def test_extreme_in_region(self):
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 0), float(5 + 5 * self.width))
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 1), float(6 + 6 * self.width))

        self.assertEqual(self.img.extreme_in_region(1, 0, 3, 4, 0), 1.0)
        self.assertEqual(self.img.extreme_in_region(1, 0, 3, 4, 1), float(3 + 4 * self.width))

        self.img.set_pixel(5, 5, KB_NO_DATA)
        self.img.set_pixel(5, 6, KB_NO_DATA)
        self.img.set_pixel(6, 6, KB_NO_DATA)
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 0), float(6 + 5 * self.width))
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 1), float(6 + 5 * self.width))

        self.img.set_pixel(6, 5, KB_NO_DATA)
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 0), KB_NO_DATA)
        self.assertEqual(self.img.extreme_in_region(5, 5, 6, 6, 1), KB_NO_DATA)

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

        # Apply masks to images 1 and 3.
        img1.apply_mask(1, [], raw_image(np.array([[0, 1], [0, 1], [0, 1]])))
        img3.apply_mask(1, [], raw_image(np.array([[0, 0], [1, 1], [1, 0]])))
        median_image2 = create_median_image([img1, img2, img3])

        self.assertEqual(median_image2.get_width(), 2)
        self.assertEqual(median_image2.get_height(), 3)
        self.assertAlmostEqual(median_image2.get_pixel(0, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(median_image2.get_pixel(1, 0), -1.0, delta=1e-6)
        self.assertAlmostEqual(median_image2.get_pixel(0, 1), 1.5, delta=1e-6)
        self.assertAlmostEqual(median_image2.get_pixel(1, 1), 3.5, delta=1e-6)
        self.assertAlmostEqual(median_image2.get_pixel(0, 2), 2.35, delta=1e-6)
        self.assertAlmostEqual(median_image2.get_pixel(1, 2), 3.15, delta=1e-6)

    def test_create_median_image_more(self):
        img1 = raw_image(np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 0.1]]))
        img2 = raw_image(np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]))
        img3 = raw_image(np.array([[3.0, -2.0], [-2.0, 5.0], [4.0, 0.3]]))
        img4 = raw_image(np.array([[4.0, 3.0], [3.0, 6.0], [5.0, 0.1]]))
        img5 = raw_image(np.array([[5.0, -3.0], [-3.0, 7.0], [7.0, 0.0]]))
        img6 = raw_image(np.array([[6.0, 2.0], [2.0, 4.0], [6.0, 0.1]]))
        img7 = raw_image(np.array([[7.0, 3.0], [3.0, 3.0], [3.0, 0.0]]))

        img1.apply_mask(1, [], raw_image(np.array([[0, 0], [1, 1], [0, 0]])))
        img2.apply_mask(1, [], raw_image(np.array([[0, 0], [1, 1], [1, 0]])))
        img3.apply_mask(1, [], raw_image(np.array([[0, 0], [0, 1], [0, 0]])))
        img4.apply_mask(1, [], raw_image(np.array([[0, 0], [0, 1], [0, 0]])))
        img5.apply_mask(1, [], raw_image(np.array([[0, 1], [0, 1], [0, 0]])))
        img6.apply_mask(1, [], raw_image(np.array([[0, 1], [1, 1], [0, 0]])))
        img7.apply_mask(1, [], raw_image(np.array([[0, 0], [1, 1], [0, 0]])))

        vect = [img1, img2, img3, img4, img5, img6, img7]
        median_image = create_median_image(vect)

        self.assertEqual(median_image.get_width(), 2)
        self.assertEqual(median_image.get_height(), 3)
        self.assertAlmostEqual(median_image.get_pixel(0, 0), 4.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(1, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(0, 1), -2.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(1, 1), 0.0, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(0, 2), 4.5, delta=1e-6)
        self.assertAlmostEqual(median_image.get_pixel(1, 2), 0.1, delta=1e-6)

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

        # Apply masks to images 1 and 3.
        img1.apply_mask(1, [], raw_image(np.array([[0, 1], [0, 1], [0, 1]])))
        img3.apply_mask(1, [], raw_image(np.array([[0, 0], [1, 1], [1, 0]])))
        summed_image2 = create_summed_image([img1, img2, img3])

        self.assertEqual(summed_image2.get_width(), 2)
        self.assertEqual(summed_image2.get_height(), 3)
        self.assertAlmostEqual(summed_image2.get_pixel(0, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(summed_image2.get_pixel(1, 0), -2.0, delta=1e-6)
        self.assertAlmostEqual(summed_image2.get_pixel(0, 1), 3.0, delta=1e-6)
        self.assertAlmostEqual(summed_image2.get_pixel(1, 1), 3.5, delta=1e-6)
        self.assertAlmostEqual(summed_image2.get_pixel(0, 2), 4.7, delta=1e-6)
        self.assertAlmostEqual(summed_image2.get_pixel(1, 2), 6.3, delta=1e-6)

    def test_create_mean_image(self):
        img1 = raw_image(np.array([[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]]))
        img2 = raw_image(np.array([[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]]))
        img3 = raw_image(np.array([[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]]))
        mean_image = create_mean_image([img1, img2, img3])

        self.assertEqual(mean_image.get_width(), 2)
        self.assertEqual(mean_image.get_height(), 3)
        self.assertAlmostEqual(mean_image.get_pixel(0, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(mean_image.get_pixel(1, 0), -1.0, delta=1e-6)
        self.assertAlmostEqual(mean_image.get_pixel(0, 1), 2.0, delta=1e-6)
        self.assertAlmostEqual(mean_image.get_pixel(1, 1), 9.5 / 3.0, delta=1e-6)
        self.assertAlmostEqual(mean_image.get_pixel(0, 2), 8.8 / 3.0, delta=1e-6)
        self.assertAlmostEqual(mean_image.get_pixel(1, 2), 9.4 / 3.0, delta=1e-6)

        # Apply masks to images 1, 2, and 3.
        img1.apply_mask(1, [], raw_image(np.array([[0, 1], [0, 1], [0, 1]])))
        img2.apply_mask(1, [], raw_image(np.array([[0, 0], [0, 0], [0, 1]])))
        img3.apply_mask(1, [], raw_image(np.array([[0, 0], [1, 1], [1, 1]])))
        mean_image2 = create_mean_image([img1, img2, img3])

        self.assertEqual(mean_image2.get_width(), 2)
        self.assertEqual(mean_image2.get_height(), 3)
        self.assertAlmostEqual(mean_image2.get_pixel(0, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(mean_image2.get_pixel(1, 0), -1.0, delta=1e-6)
        self.assertAlmostEqual(mean_image2.get_pixel(0, 1), 1.5, delta=1e-6)
        self.assertAlmostEqual(mean_image2.get_pixel(1, 1), 3.5, delta=1e-6)
        self.assertAlmostEqual(mean_image2.get_pixel(0, 2), 2.35, delta=1e-6)
        self.assertAlmostEqual(mean_image2.get_pixel(1, 2), 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
