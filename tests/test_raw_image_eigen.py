import tempfile
import unittest

import numpy as np
import timeit
from kbmod.search import RawImageEigen as RawImage
from kbmod.search import (
    HAS_GPU,
    KB_NO_DATA,
    PSF,
    create_median_image_eigen,
    create_summed_image_eigen,
    create_mean_image_eigen,
)


class test_RawImage(unittest.TestCase):
    def setUp(self, width=10, height=12):
        self.width = width
        self.height = height

        # self.const_arr =  10.0 * np.ones(height, width, dtype=np.single)
        self.array = np.arange(0, width * height, dtype=np.single).reshape(height, width)

        self.masked_array = 10.0 * np.ones((height, width), dtype=np.single)
        self.masked_array[5, 6] = 0.1
        self.masked_array[5, 7] = KB_NO_DATA
        self.masked_array[3, 1] = 100.0
        self.masked_array[4, 4] = KB_NO_DATA
        self.masked_array[5, 5] = 100.0

    def test_create(self):
        """Test RawImage constructors."""
        # Default constructor
        img = RawImage()
        self.assertEqual(img.width, 0)
        self.assertEqual(img.height, 0)
        self.assertEqual(img.obstime, -1.0)

        # from NumPy arrays
        img = RawImage(img=self.array, obs_time=10.0)
        self.assertEqual(img.image.shape, (self.height, self.width))
        self.assertEqual(img.obstime, 10.0)
        self.assertEqual(img.npixels, self.width * self.height)
        self.assertTrue((img.image == self.array).all())

        img2 = RawImage(img=self.array)
        self.assertTrue((img2.image == img.image).all())
        self.assertEqual(img2.obstime, -1.0)

        # from dimensions
        img = RawImage(self.height, self.width)
        self.assertEqual(img.image.shape, (self.height, self.width))
        self.assertEqual(img.obstime, -1.0)
        self.assertTrue((img.image == 0).all())

        # dimensions and optional values
        img = RawImage(self.height, self.width, 10)
        self.assertTrue((img.image == 10).all())

        img = RawImage(self.height, self.width, 10, 12.5)
        self.assertTrue((img.image == 10).all())
        self.assertEqual(img.obstime, 12.5)

        img = RawImage(self.height, self.width, value=7.5, obs_time=12.5)
        self.assertTrue((img.image == 7.5).all())
        self.assertEqual(img.obstime, 12.5)

        # copy constructor, set the old image to all zeros and change the time.
        img = RawImage(img=self.array, obs_time=10.0)
        img2 = RawImage(img)
        img.set_all(0.0)
        img.obstime = 1.0
        self.assertTrue((img2.image == self.array).all())
        self.assertEqual(img2.obstime, 10.0)

    def test_pixel_getters(self):
        """Test RawImage masked pixel value getters"""
        img = RawImage(img=self.array, obs_time=10.0)
        self.assertEqual(img.get_pixel(-1, 5), KB_NO_DATA)
        self.assertEqual(img.get_pixel(5, self.width), KB_NO_DATA)
        self.assertEqual(img.get_pixel(5, -1), KB_NO_DATA)
        self.assertEqual(img.get_pixel(self.height, 5), KB_NO_DATA)

    def test_approx_equal(self):
        """Test RawImage pixel value setters."""
        img = RawImage(img=self.array, obs_time=10.0)

        # This test is testing L^\infy norm closeness. Eigen isApprox uses L2
        # norm closeness.
        img2 = RawImage(img)
        img2.imref += 0.0001
        self.assertTrue(np.allclose(img.image, img2.image, atol=0.01))

        # Add a single NO_DATA entry.
        img.set_pixel(5, 7, KB_NO_DATA)
        self.assertFalse(np.allclose(img.image, img2.image, atol=0.01))

        img2.set_pixel(5, 7, KB_NO_DATA)
        self.assertTrue(np.allclose(img.image, img2.image, atol=0.01))

        # Add a second NO_DATA entry to image 2.
        img2.set_pixel(7, 7, KB_NO_DATA)
        self.assertFalse(np.allclose(img.image, img2.image, atol=0.01))

        img.set_pixel(7, 7, KB_NO_DATA)
        self.assertTrue(np.allclose(img.image, img2.image, atol=0.01))

        # Add some noise to mess up an observation.
        img2.set_pixel(1, 3, 13.1)  # img.image[1, 3]+0.1)
        self.assertFalse(np.allclose(img.image, img2.image, atol=0.01))

        # test set_all
        img.set_all(15.0)
        self.assertTrue((img.image == 15).all())

    def test_get_bounds(self):
        """Test RawImage masked min/max bounds."""
        img = RawImage(self.masked_array)
        lower, upper = img.compute_bounds()
        self.assertAlmostEqual(lower, 0.1, delta=1e-6)
        self.assertAlmostEqual(upper, 100.0, delta=1e-6)

    def test_find_peak(self):
        "Test RawImage find_peak"
        img = RawImage(self.masked_array)
        idx = img.find_peak(False)
        self.assertEqual(idx.i, 5)
        self.assertEqual(idx.j, 5)

        # We found the peak furthest to the center.
        idx = img.find_peak(True)
        self.assertEqual(idx.i, 3)
        self.assertEqual(idx.j, 1)

    def test_find_central_moments(self):
        """Test RawImage central moments."""
        img = RawImage(5, 5, value=0.1)

        # Try something mostly symmetric and centered.
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

    def convolve_psf_identity(self, device):
        psf_data = np.zeros((3, 3), dtype=np.single)
        psf_data[1, 1] = 1.0
        p = PSF(psf_data)

        img = RawImage(self.array)

        if device.upper() == "CPU":
            img.convolve_cpu(p)
        elif device.upper() == "GPU":
            img.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        self.assertTrue(np.allclose(self.array, img.image, 0.0001))

    def test_convolve_psf_identity_cpu(self):
        """Test convolution with a identity kernel on CPU"""
        self.convolve_psf_identity("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_identity_gpu(self):
        """Test convolution with a identity kernel on GPU"""
        self.convolve_psf_identity("GPU")

    def convolve_psf_mask(self, device):
        p = PSF(1.0)

        # Mask out three pixels.
        img = RawImage(self.array)
        img.set_pixel(0, 3, KB_NO_DATA)
        img.set_pixel(5, 6, KB_NO_DATA)
        img.set_pixel(5, 7, KB_NO_DATA)

        if device.upper() == "CPU":
            img.convolve_cpu(p)
        elif device.upper() == "GPU":
            img.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        # Check that the same pixels are masked.
        for x in range(self.width):
            for y in range(self.height):
                if (y == 5 and x == 6) or (y == 0 and x == 3) or (y == 5 and x == 7):
                    self.assertFalse(img.pixel_has_data(y, x))
                else:
                    self.assertTrue(img.pixel_has_data(y, x))

    def test_convolve_psf_mask_cpu(self):
        """Test masked convolution with a identity kernel on CPU"""
        self.convolve_psf_mask("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_mask_gpu(self):
        """Test masked convolution with a identity kernel on GPU"""
        self.convolve_psf_mask("CPU")

    # confused, sort out later
    def convolve_psf_average(self, device):
        # Mask out a single pixel.
        img = RawImage(self.array)
        img.set_pixel(4, 6, KB_NO_DATA)

        # Set up a simple "averaging" psf to convolve.
        psf_data = np.zeros((5, 5), dtype=np.single)
        psf_data[1:4, 1:4] = 0.1111111
        p = PSF(psf_data)
        self.assertAlmostEqual(p.get_sum(), 1.0, delta=0.00001)

        img2 = RawImage(img)
        if device.upper() == "CPU":
            img2.convolve_cpu(p)
        elif device.upper() == "GPU":
            img2.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        for x in range(self.width):
            for y in range(self.height):
                # Compute the weighted average around (x, y)
                # in the original image.
                running_sum = 0.0
                count = 0.0
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        value = img.get_pixel(y + j, x + i)
                        psf_value = 0.1111111
                        if i == -2 or i == 2 or j == -2 or j == 2:
                            psf_value = 0.0

                        if value != KB_NO_DATA:
                            running_sum += psf_value * value
                            count += psf_value
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                if x == 6 and y == 4:
                    self.assertFalse(img2.pixel_has_data(y, x))
                else:
                    self.assertAlmostEqual(img2.get_pixel(y, x), ave, delta=0.001)

    def test_convolve_psf_average(self):
        """Test convolution on CPU produces expected values."""
        self.convolve_psf_average("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_average_gpu(self):
        """Test convolution on GPU produces expected values."""
        self.convolve_psf_average("GPU")

    def convolve_psf_orientation_cpu(self, device):
        """Test convolution on CPU with a non-symmetric PSF"""
        img = RawImage(self.array.copy())

        # Set up a non-symmetric psf where orientation matters.
        psf_data = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.4], [0.0, 0.1, 0.0]]
        p = PSF(np.array(psf_data))

        img2 = RawImage(img)
        if device.upper() == "CPU":
            img2.convolve_cpu(p)
        elif device.upper() == "GPU":
            img2.convolve_gpu(p)
        else:
            raise ValueError(f"Unknown device. Expected GPU or CPU got {device}")

        for x in range(img.width):
            for y in range(img.height):
                running_sum = 0.5 * img.get_pixel(y, x)
                count = 0.5
                if img.pixel_has_data(y, x + 1):
                    running_sum += 0.4 * img.get_pixel(y, x + 1)
                    count += 0.4
                if img.pixel_has_data(y + 1, x):
                    running_sum += 0.1 * img.get_pixel(y + 1, x)
                    count += 0.1
                ave = running_sum / count

                # Compute the manually computed result with the convolution.
                self.assertAlmostEqual(img2.get_pixel(y, x), ave, delta=0.001)

    def test_convolve_psf_orientation_cpu(self):
        """Test convolution on CPU with a non-symmetric PSF"""
        self.convolve_psf_orientation_cpu("CPU")

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_convolve_psf_orientation_gpu(self):
        """Test convolution on GPU with a non-symmetric PSF"""
        self.convolve_psf_orientation_cpu("GPU")

    def test_grow_mask(self):
        """Test grow_mask based on manhattan distances."""
        img = RawImage(self.array)
        img.set_pixel(5, 7, KB_NO_DATA)
        img.set_pixel(3, 7, KB_NO_DATA)

        for y in range(img.height):
            for x in range(img.width):
                should_mask = (y == 3 and x == 7) or (y == 5 and x == 7)
                self.assertEqual(img.pixel_has_data(y, x), not should_mask)

        # Grow the mask by one pixel.
        img.grow_mask(1)
        for y in range(img.height):
            for x in range(img.width):
                dist = min([abs(7 - x) + abs(3 - y), abs(7 - x) + abs(5 - y)])
                self.assertEqual(img.pixel_has_data(y, x), dist > 1)

        # Grow the mask by an additional two pixels (for a total of 3).
        img.grow_mask(2)
        for y in range(img.height):
            for x in range(img.width):
                dist = min([abs(7 - x) + abs(3 - y), abs(7 - x) + abs(5 - y)])
                self.assertEqual(img.pixel_has_data(y, x), dist > 3)

    def test_make_stamp(self):
        """Test stamp creation."""
        img = RawImage(self.array)
        stamp = img.create_stamp(2.5, 2.5, 2, True, False)
        self.assertEqual(stamp.shape, (5, 5))
        self.assertTrue((stamp == self.array[0:5, 0:5]).all())

    def test_read_write_file(self):
        """Test file writes and reads correctly."""
        img = RawImage(self.array, 10.0)
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_RawImage"
            full_path = "%s/%s.fits" % (dir_name, file_name)

            img.save_fits(full_path)

            # Reload the file.
            img2 = RawImage(0, 0)
            img2.load_fits(full_path, 0)
            self.assertEqual(img2.width, self.width)
            self.assertEqual(img2.height, self.height)
            self.assertEqual(img2.npixels, self.width * self.height)
            self.assertEqual(img2.obstime, 10.0)
            self.assertTrue(np.allclose(img.image, img2.image, atol=1e-5))

    def test_stack_file(self):
        """Test multi-extension FITS files write and read correctly."""
        img = RawImage(self.array, 10.0)
        with tempfile.TemporaryDirectory() as dir_name:
            file_name = "tmp_RawImage"
            full_path = "%s/%s.fits" % (dir_name, file_name)

            # Save the image and create a file.
            img.save_fits(full_path)

            # Add 4 more layers at different times.
            for i in range(1, 5):
                img.obstime = 10.0 + 2.0 * i
                img.append_fits_extension(full_path)

            # Check that we get 5 layers with the correct times.
            img2 = RawImage(0, 0)
            for i in range(5):
                img2.load_fits(full_path, i)

                self.assertEqual(img2.width, self.width)
                self.assertEqual(img2.height, self.height)
                self.assertEqual(img2.npixels, self.width * self.height)
                self.assertEqual(img2.obstime, 10.0 + 2.0 * i)
                self.assertTrue(np.allclose(img.image, img2.image, 1e-5))

    def test_create_median_image(self):
        """Tests median image coaddition."""
        arrs = np.array(
            [
                [[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]],
                [[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]],
                [[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]],
            ],
            dtype=np.single,
        )
        imgs = list(map(RawImage, arrs))

        median_image = create_median_image_eigen(imgs)

        expected = np.median(arrs, axis=0)
        self.assertEqual(median_image.width, 2)
        self.assertEqual(median_image.height, 3)
        self.assertTrue(np.allclose(median_image.image, expected, atol=1e-6))

        # Apply masks to images 1 and 3.
        imgs[0].apply_mask(1, [], RawImage(np.array([[0, 1], [0, 1], [0, 1]], dtype=np.single)))
        imgs[2].apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [1, 0]], dtype=np.single)))

        median_image = create_median_image_eigen(imgs)

        expected = np.array([[0, -1], [1.5, 3.5], [2.35, 3.15]], dtype=np.single)
        self.assertEqual(median_image.width, 2)
        self.assertEqual(median_image.height, 3)
        self.assertTrue(np.allclose(median_image.image, expected, atol=1e-6))

        # More median image tests
        arrs = np.array(
            [
                [[1.0, -1.0], [-1.0, 1.0], [1.0, 0.1]],
                [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
                [[3.0, -2.0], [-2.0, 5.0], [4.0, 0.3]],
                [[4.0, 3.0], [3.0, 6.0], [5.0, 0.1]],
                [[5.0, -3.0], [-3.0, 7.0], [7.0, 0.0]],
                [[6.0, 2.0], [2.0, 4.0], [6.0, 0.1]],
                [[7.0, 3.0], [3.0, 3.0], [3.0, 0.0]],
            ],
            dtype=np.single,
        )

        masks = np.array(
            [
                np.array([[0, 0], [1, 1], [0, 0]]),
                np.array([[0, 0], [1, 1], [1, 0]]),
                np.array([[0, 0], [0, 1], [0, 0]]),
                np.array([[0, 0], [0, 1], [0, 0]]),
                np.array([[0, 1], [0, 1], [0, 0]]),
                np.array([[0, 1], [1, 1], [0, 0]]),
                np.array([[0, 0], [1, 1], [0, 0]]),
            ],
            dtype=np.single,
        )

        imgs = list(map(RawImage, arrs))
        for img, mask in zip(imgs, masks):
            img.apply_mask(1, [], RawImage(mask))

        median_image = create_median_image_eigen(imgs)
        expected = np.array([[4, 0], [-2, 0], [4.5, 0.1]], dtype=np.single)
        self.assertEqual(median_image.width, 2)
        self.assertEqual(median_image.height, 3)
        self.assertTrue(np.allclose(median_image.image, expected, atol=1e-6))

    def test_create_summed_image(self):
        arrs = np.array(
            [
                [[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]],
                [[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]],
                [[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]],
            ],
            dtype=np.single,
        )
        imgs = list(map(RawImage, arrs))

        summed_image = create_summed_image_eigen(imgs)

        expected = arrs.sum(axis=0)
        self.assertEqual(summed_image.width, 2)
        self.assertEqual(summed_image.height, 3)
        self.assertTrue(np.allclose(expected, summed_image.image, atol=1e-6))

        # Apply masks to images 1 and 3.
        imgs[0].apply_mask(1, [], RawImage(np.array([[0, 1], [0, 1], [0, 1]], dtype=np.single)))
        imgs[2].apply_mask(1, [], RawImage(np.array([[0, 0], [1, 1], [1, 0]], dtype=np.single)))

        summed_image = create_summed_image_eigen(imgs)

        expected = np.array([[0, -2], [3, 3.5], [4.7, 6.3]], dtype=np.single)
        self.assertEqual(summed_image.width, 2)
        self.assertEqual(summed_image.height, 3)
        self.assertTrue(np.allclose(expected, summed_image.image, atol=1e-6))

    def test_create_mean_image(self):
        arrs = np.array(
            [
                [[0.0, -1.0], [2.0, 1.0], [0.7, 3.1]],
                [[1.0, 0.0], [1.0, 3.5], [4.0, 3.0]],
                [[-1.0, -2.0], [3.0, 5.0], [4.1, 3.3]],
            ],
            dtype=np.single,
        )
        imgs = list(map(RawImage, arrs))

        mean_image = create_mean_image_eigen(imgs)

        expected = arrs.mean(axis=0)
        self.assertEqual(mean_image.width, 2)
        self.assertEqual(mean_image.height, 3)
        self.assertTrue(np.allclose(mean_image.image, expected, atol=1e-6))

        # Apply masks to images 1, 2, and 3.
        masks = np.array(
            [[[0, 1], [0, 1], [0, 1]], [[0, 0], [0, 0], [0, 1]], [[0, 0], [1, 1], [1, 1]]], dtype=np.single
        )
        for img, mask in zip(imgs, masks):
            img.apply_mask(1, [], RawImage(mask))

        mean_image = create_mean_image_eigen(imgs)

        expected = np.array([[0, -1], [1.5, 3.5], [2.35, 0]], dtype=np.single)
        self.assertEqual(mean_image.width, 2)
        self.assertEqual(mean_image.height, 3)
        self.assertTrue(np.allclose(mean_image.image, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
