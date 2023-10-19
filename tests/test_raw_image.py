import tempfile
import unittest

import numpy as np

from kbmod.search import *


class test_RawImage(unittest.TestCase):
    def setUp(self):
        self.width = 10
        self.height = 12
        self.img = RawImage(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.img.set_pixel(x, y, float(x + y * self.width))
        self.img.set_obstime(10.0)

    def test_create(self):
        self.assertEqual(self.img.get_width(), self.width)
        self.assertEqual(self.img.get_height(), self.height)
        self.assertEqual(self.img.get_npixels(), self.width * self.height)
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(self.img.pixel_has_data(x, y))
                self.assertEqual(self.img.get_pixel(x, y), float(x + y * self.width))
        self.assertEqual(self.img.get_obstime(), 10.0)

        # Pixels outside the image have no data.
        self.assertFalse(self.img.pixel_has_data(-1, 5))
        self.assertFalse(self.img.pixel_has_data(self.width, 5))
        self.assertFalse(self.img.pixel_has_data(5, -1))
        self.assertFalse(self.img.pixel_has_data(5, self.height))

        self.assertEqual(self.img.get_pixel(-1, 5), KB_NO_DATA)
        self.assertEqual(self.img.get_pixel(self.width, 5), KB_NO_DATA)
        self.assertEqual(self.img.get_pixel(5, -1), KB_NO_DATA)
        self.assertEqual(self.img.get_pixel(5, self.height), KB_NO_DATA)

    def test_approx_equal(self):
        # Make approximate copy.
        img2 = RawImage(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                img2.set_pixel(x, y, self.img.get_pixel(x, y) + 0.0001)
        self.assertTrue(self.img.approx_equal(img2, 0.01))

        # Add a single NO_DATA entry.
        self.img.set_pixel(5, 7, KB_NO_DATA)
        self.assertFalse(self.img.approx_equal(img2, 0.01))
        img2.set_pixel(5, 7, KB_NO_DATA)
        self.assertTrue(self.img.approx_equal(img2, 0.01))

        # Add a second NO_DATA entry to image 2.
        img2.set_pixel(7, 7, KB_NO_DATA)
        self.assertFalse(self.img.approx_equal(img2, 0.01))
        self.img.set_pixel(7, 7, KB_NO_DATA)
        self.assertTrue(self.img.approx_equal(img2, 0.01))

        # Add some noise to mess up an observation.
        img2.set_pixel(1, 3, self.img.get_pixel(1, 3) + 0.1)
        self.assertFalse(self.img.approx_equal(img2, 0.01))

    def test_copy(self):
        # Copy the image.
        img2 = RawImage(self.img)
        self.assertEqual(img2.get_width(), self.width)
        self.assertEqual(img2.get_height(), self.height)
        self.assertEqual(img2.get_npixels(), self.width * self.height)
        self.assertTrue(self.img.approx_equal(img2, 0.0001))
        self.assertEqual(img2.get_obstime(), 10.0)

        # Set the old image to all zeros and change the time.
        self.img.set_all(0.0)
        self.img.set_obstime(1.0)

        # Check the new image is still set correctly.
        for x in range(self.width):
            for y in range(self.height):
                self.assertTrue(img2.pixel_has_data(x, y))
                self.assertEqual(img2.get_pixel(x, y), float(x + y * self.width))
        self.assertEqual(img2.get_obstime(), 10.0)

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

    def test_find_peak(self):
        self.img.set_all(10.0)
        self.img.set_pixel(5, 6, 0.1)
        self.img.set_pixel(5, 7, KB_NO_DATA)
        self.img.set_pixel(3, 1, 100.0)
        self.img.set_pixel(4, 4, KB_NO_DATA)
        self.img.set_pixel(5, 5, KB_NO_DATA)

        peak = self.img.find_peak(False)
        self.assertEqual(int(peak.x), 3)
        self.assertEqual(int(peak.y), 1)

    def test_find_peak_duplicate(self):
        self.img.set_all(10.0)
        self.img.set_pixel(5, 6, 0.1)
        self.img.set_pixel(5, 7, KB_NO_DATA)
        self.img.set_pixel(3, 1, 100.0)
        self.img.set_pixel(4, 4, KB_NO_DATA)
        self.img.set_pixel(5, 5, 100.0)

        # We found the peak closest to the center.
        peak = self.img.find_peak(False)
        self.assertEqual(int(peak.x), 5)
        self.assertEqual(int(peak.y), 5)

        # We found the peak furthest to the center.
        peak = self.img.find_peak(True)
        self.assertEqual(int(peak.x), 3)
        self.assertEqual(int(peak.y), 1)

    def test_find_central_moments(self):
        img = RawImage(5, 5)

        # Try something mostly symmetric and centered.
        img.set_all(0.1)
        img.set_pixel(2, 2, 10.0)
        img.set_pixel(2, 1, 5.0)
        img.set_pixel(1, 2, 5.0)
        img.set_pixel(2, 3, 5.0)
        img.set_pixel(3, 2, 5.0)

        img_mom = img.find_central_moments()
        self.assertAlmostEqual(img_mom.m00, 1.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m01, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m10, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m11, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m02, 0.3322, delta=1e-4)
        self.assertAlmostEqual(img_mom.m20, 0.3322, delta=1e-4)

        # Try something flat symmetric and centered.
        img.set_all(2.0)
        img_mom = img.find_central_moments()

        self.assertAlmostEqual(img_mom.m00, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m01, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m10, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m11, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m02, 0.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m20, 0.0, delta=1e-4)

        # Try something with a few non-symmetric peaks.
        img.set_all(0.4)
        img.set_pixel(2, 2, 5.0)
        img.set_pixel(0, 1, 5.0)
        img.set_pixel(3, 3, 10.0)
        img.set_pixel(0, 3, 0.2)
        img_mom = img.find_central_moments()

        self.assertAlmostEqual(img_mom.m00, 1.0, delta=1e-4)
        self.assertAlmostEqual(img_mom.m01, 0.20339, delta=1e-4)
        self.assertAlmostEqual(img_mom.m10, 0.03390, delta=1e-4)
        self.assertAlmostEqual(img_mom.m11, 0.81356, delta=1e-4)
        self.assertAlmostEqual(img_mom.m02, 1.01695, delta=1e-4)
        self.assertAlmostEqual(img_mom.m20, 1.57627, delta=1e-4)

    def test_convolve_psf_identity_cpu(self):
        psf_data = [[0.0 for _ in range(3)] for _ in range(3)]
        psf_data[1][1] = 1.0
        p = PSF(np.array(psf_data))

        img2 = RawImage(self.img)
        img2.convolve_cpu(p)

        # Check that the image is unchanged.
        self.assertTrue(self.img.approx_equal(img2, 0.0001))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_identity_gpu(self):
        psf_data = [[0.0 for _ in range(3)] for _ in range(3)]
        psf_data[1][1] = 1.0
        p = PSF(np.array(psf_data))

        img2 = RawImage(self.img)
        img2.convolve(p)

        # Check that the image is unchanged.
        self.assertTrue(self.img.approx_equal(img2, 0.0001))

    def test_convolve_psf_mask_cpu(self):
        p = PSF(1.0)

        # Mask out three pixels.
        self.img.set_pixel(5, 6, KB_NO_DATA)
        self.img.set_pixel(0, 3, KB_NO_DATA)
        self.img.set_pixel(5, 7, KB_NO_DATA)

        img2 = RawImage(self.img)
        img2.convolve_cpu(p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (x == 5 and y == 6) or (x == 0 and y == 3) or (x == 5 and y == 7):
                    self.assertFalse(img2.pixel_has_data(x, y))
                else:
                    self.assertTrue(img2.pixel_has_data(x, y))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_mask_gpu(self):
        p = PSF(1.0)

        # Mask out three pixels.
        self.img.set_pixel(5, 6, KB_NO_DATA)
        self.img.set_pixel(0, 3, KB_NO_DATA)
        self.img.set_pixel(5, 7, KB_NO_DATA)

        img2 = RawImage(self.img)
        img2.convolve(p)

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (x == 5 and y == 6) or (x == 0 and y == 3) or (x == 5 and y == 7):
                    self.assertFalse(img2.pixel_has_data(x, y))
                else:
                    self.assertTrue(img2.pixel_has_data(x, y))

    def test_convolve_psf_average_cpu(self):
        # Mask out a single pixel.
        self.img.set_pixel(6, 4, KB_NO_DATA)

        # Set up a simple "averaging" psf to convolve.
        psf_data = [[0.0 for _ in range(5)] for _ in range(5)]
        for x in range(1, 4):
            for y in range(1, 4):
                psf_data[x][y] = 0.1111111
        p = PSF(np.array(psf_data))
        self.assertAlmostEqual(p.get_sum(), 1.0, delta=0.00001)

        img2 = RawImage(self.img)
        img2.convolve_cpu(p)

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
                if x == 6 and y == 4:
                    self.assertFalse(img2.pixel_has_data(x, y))
                else:
                    self.assertAlmostEqual(img2.get_pixel(x, y), ave, delta=0.001)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_average_gpu(self):
        # Mask out a single pixel.
        self.img.set_pixel(6, 4, KB_NO_DATA)

        # Set up a simple "averaging" psf to convolve.
        psf_data = [[0.0 for _ in range(5)] for _ in range(5)]
        for x in range(1, 4):
            for y in range(1, 4):
                psf_data[x][y] = 0.1111111
        p = PSF(np.array(psf_data))
        self.assertAlmostEqual(p.get_sum(), 1.0, delta=0.00001)

        img2 = RawImage(self.img)
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
                if x == 6 and y == 4:
                    self.assertFalse(img2.pixel_has_data(x, y))
                else:
                    self.assertAlmostEqual(img2.get_pixel(x, y), ave, delta=0.001)

    def test_convolve_psf_orientation_cpu(self):
        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = PSF(np.array(psf_data))

        img2 = RawImage(self.img)
        img2.convolve_cpu(p)

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

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_orientation_gpu(self):
        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = PSF(np.array(psf_data))

        img2 = RawImage(self.img)
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

    def test_grow_mask(self):
        self.img.set_pixel(5, 7, KB_NO_DATA)
        self.img.set_pixel(3, 7, KB_NO_DATA)

        for y in range(self.img.get_height()):
            for x in range(self.img.get_width()):
                should_mask = (x == 3 and y == 7) or (x == 5 and y == 7)
                self.assertEqual(self.img.pixel_has_data(x, y), not should_mask)

        # Grow the mask by one pixel.
        self.img.grow_mask(1)
        for y in range(self.img.get_height()):
            for x in range(self.img.get_width()):
                dist = min([abs(3 - x) + abs(7 - y), abs(5 - x) + abs(7 - y)])
                self.assertEqual(self.img.pixel_has_data(x, y), dist > 1)

        # Grow the mask by an additional two pixels (for a total of 3).
        self.img.grow_mask(2)
        for y in range(self.img.get_height()):
            for x in range(self.img.get_width()):
                dist = min([abs(3 - x) + abs(7 - y), abs(5 - x) + abs(7 - y)])
                self.assertEqual(self.img.pixel_has_data(x, y), dist > 3)

    def test_make_stamp(self):
        for x in range(self.width):
            for y in range(self.height):
                self.img.set_pixel(x, y, float(x + y * self.width))

        stamp = self.img.create_stamp(2.5, 2.5, 2, False)
        self.assertEqual(stamp.get_height(), 5)
        self.assertEqual(stamp.get_width(), 5)
        for x in range(-2, 3):
            for y in range(-2, 3):
                self.assertAlmostEqual(
                    stamp.get_pixel(2 + x, 2 + y), float((x + 2) + (y + 2) * self.width), delta=0.001
                )

        # Check that the stamp has the same obstime.
        self.assertEqual(stamp.get_obstime(), 10.0)

    def test_read_write_file(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_RawImage"
            full_path = "%s/%s.fits" % (dir_name, file_name)

            self.img.save_fits(full_path)

            # Reload the file.
            img2 = RawImage(0, 0)
            img2.load_fits(full_path, 0)
            self.assertEqual(img2.get_width(), self.width)
            self.assertEqual(img2.get_height(), self.height)
            self.assertEqual(img2.get_npixels(), self.width * self.height)
            self.assertEqual(img2.get_obstime(), 10.0)
            self.assertTrue(self.img.approx_equal(img2, 1e-5))

    def test_stack_file(self):
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_RawImage"
            full_path = "%s/%s.fits" % (dir_name, file_name)

            # Save the image and create a file.
            self.img.save_fits(full_path)

            # Add 4 more layers at different times.
            for i in range(1, 5):
                self.img.set_obstime(10.0 + 2.0 * i)
                self.img.append_fits_layer(full_path)

            # Check that we get 5 layers with the correct times.
            img2 = RawImage(0, 0)
            for i in range(5):
                img2.load_fits(full_path, i)

                self.assertEqual(img2.get_width(), self.width)
                self.assertEqual(img2.get_height(), self.height)
                self.assertEqual(img2.get_npixels(), self.width * self.height)
                self.assertEqual(img2.get_obstime(), 10.0 + 2.0 * i)
                self.assertTrue(self.img.approx_equal(img2, 1e-5))

    def test_create_median_image(self):
        img1 = RawImage(np.array([[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]]))
        img2 = RawImage(np.array([[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]]))
        img3 = RawImage(np.array([[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]]))
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
        img1.apply_mask(1, [], RawImage(np.array([[0, 1], [0, 1], [0, 1]])))
        img3.apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [1, 0]])))
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
        img1 = RawImage(np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, 0.1]]))
        img2 = RawImage(np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]))
        img3 = RawImage(np.array([[3.0, -2.0], [-2.0, 5.0], [4.0, 0.3]]))
        img4 = RawImage(np.array([[4.0, 3.0], [3.0, 6.0], [5.0, 0.1]]))
        img5 = RawImage(np.array([[5.0, -3.0], [-3.0, 7.0], [7.0, 0.0]]))
        img6 = RawImage(np.array([[6.0, 2.0], [2.0, 4.0], [6.0, 0.1]]))
        img7 = RawImage(np.array([[7.0, 3.0], [3.0, 3.0], [3.0, 0.0]]))

        img1.apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [0, 0]])))
        img2.apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [1, 0]])))
        img3.apply_mask(1, [], RawImage(np.array([[0, 0], [0, 1], [0, 0]])))
        img4.apply_mask(1, [], RawImage(np.array([[0, 0], [0, 1], [0, 0]])))
        img5.apply_mask(1, [], RawImage(np.array([[0, 1], [0, 1], [0, 0]])))
        img6.apply_mask(1, [], RawImage(np.array([[0, 1], [1, 1], [0, 0]])))
        img7.apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [0, 0]])))

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
        img1 = RawImage(np.array([[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]]))
        img2 = RawImage(np.array([[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]]))
        img3 = RawImage(np.array([[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]]))
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
        img1.apply_mask(1, [], RawImage(np.array([[0, 1], [0, 1], [0, 1]])))
        img3.apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [1, 0]])))
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
        img1 = RawImage(np.array([[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]]))
        img2 = RawImage(np.array([[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]]))
        img3 = RawImage(np.array([[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]]))
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
        img1.apply_mask(1, [], RawImage(np.array([[0, 1], [0, 1], [0, 1]])))
        img2.apply_mask(1, [], RawImage(np.array([[0, 0], [0, 0], [0, 1]])))
        img3.apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [1, 1]])))
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
