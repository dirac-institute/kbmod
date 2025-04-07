import math
import numpy as np
import unittest

from kbmod.core.psf import PSF


class test_PSF(unittest.TestCase):
    def test_make_noop(self):
        psf0 = PSF(0.000001)
        self.assertEqual(psf0.kernel.shape, (1, 1))
        self.assertAlmostEqual(psf0.kernel[0][0], 1.0)
        self.assertEqual(psf0.width, 1)
        self.assertEqual(psf0.radius, 0)

    def test_make_invalid(self):
        # Raise an error if creating a PSF with a negative stdev.
        self.assertRaises(ValueError, PSF, -1.0)

    def test_make_from_array(self):
        arr = np.full((3, 3), 1.0 / 9.0)
        psf_arr = PSF(arr)
        self.assertEqual(psf_arr.width, 3)
        self.assertEqual(psf_arr.radius, 1)

        # We get an error if we include a NaN.
        arr[0][0] = math.nan
        self.assertRaises(ValueError, PSF, arr)

        # We get an error if we include a inf.
        arr[0][0] = math.inf
        self.assertRaises(ValueError, PSF, arr)

    # Test that the PSF sums to close to 1
    def test_from_gaussian(self):
        for std_val in range(1, 10):
            p = PSF(std_val / 5 + 0.2)
            self.assertGreater(np.sum(p.kernel), 0.95)

    def test_square(self):
        for std_val in range(1, 10):
            p = PSF(std_val / 5 + 0.2)

            # Create a square of the PSF.
            x = p.make_square()
            self.assertTrue(np.not_equal(x.kernel, p.kernel).any())

            # Squaring the PSF should not change any of the parameters.
            self.assertEqual(x.width, p.width)
            self.assertEqual(x.radius, p.radius)

    def test_convolve_psf_identity(self):
        psf_data = np.zeros((3, 3), dtype=np.single)
        psf_data[1, 1] = 1.0
        p = PSF(psf_data)

        img = np.arange(24, dtype=np.single).reshape((4, 6))
        img[1, 1] = np.nan
        img[2, 3] = np.nan
        valid_mask = np.isfinite(img)

        img2 = p.convolve_image(img)
        self.assertTrue(np.array_equal(valid_mask, np.isfinite(img2)))
        self.assertTrue(np.allclose(img[valid_mask], img2[valid_mask], 0.0001))
        self.assertTrue(np.isnan(img2[1, 1]))
        self.assertTrue(np.isnan(img2[2, 3]))

    def test_convolve_given(self):
        kernel = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.6, 0.1],
                [0.0, 0.1, 0.0],
            ]
        )
        p = PSF(kernel)
        self.assertAlmostEqual(np.sum(p.kernel), 1.0, delta=0.00001)

        # Create a fake image with a single masked value.
        img = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, np.nan, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=np.single,
        )
        orig_img = img.copy()
        valid_mask = np.isfinite(img)

        # The expected result of the convolution.
        expected = np.array(
            [
                [0.5, 1.3, 1.6, 2.7],
                [3.7, 4.4, np.nan, 5.6],
                [6.1, 7.7, 8.0, 8.3],
            ],
            dtype=np.single,
        )

        # Do the convolution check that the original image is unchanged and
        # the convolution result is as expected.
        img2 = p.convolve_image(img)
        self.assertTrue(np.array_equal(valid_mask, np.isfinite(img2)))
        self.assertTrue(np.allclose(img[valid_mask], orig_img[valid_mask], 0.0001))
        self.assertTrue(np.allclose(expected[valid_mask], img2[valid_mask], 0.0001))
        self.assertTrue(np.isnan(img2[1, 2]))

        # Do the convolution in place and check that the original image is
        # changed and the convolution result is as expected.
        img3 = p.convolve_image(img, in_place=True)
        self.assertTrue(np.array_equal(valid_mask, np.isfinite(img3)))
        self.assertTrue(np.allclose(expected[valid_mask], img[valid_mask], 0.0001))
        self.assertTrue(np.allclose(expected[valid_mask], img3[valid_mask], 0.0001))
        self.assertTrue(np.isnan(img3[1, 2]))


if __name__ == "__main__":
    unittest.main()
