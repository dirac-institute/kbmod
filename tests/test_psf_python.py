import math
import numpy as np
import unittest

from kbmod.psf import PSF


class test_PSF(unittest.TestCase):
    def test_make_simple(self):
        """Make a kernel that has equal weights."""
        kernel = np.full((5, 5), 1.0 / 25.0)
        psf = PSF(kernel)
        self.assertEqual(psf.width, 5)
        self.assertEqual(psf.radius, 2)
        self.assertEqual(psf.shape, (5, 5))
        self.assertTrue(np.allclose(psf.kernel, 1.0 / 25.0))

        # We fail if any of the kernel values are negative.
        kernel[0][0] = -1.0
        self.assertRaises(ValueError, PSF, kernel)

        # We fail if any of the kernel values are NaN.
        kernel[0][0] = math.nan
        self.assertRaises(ValueError, PSF, kernel)

        # We fail if any of the kernel values are inf.
        kernel[0][0] = math.inf
        self.assertRaises(ValueError, PSF, kernel)

        # We fail if the trying to create a kernel that is not 2D.
        # Raise an error if creating a PSF with a non-2D kernel.
        self.assertRaises(ValueError, PSF, np.ones(3))
        self.assertRaises(ValueError, PSF, np.ones((3, 3, 3)))

        # We fail if we try to create a non-square kernel or one without
        # the same radius on both sides of the center.
        self.assertRaises(ValueError, PSF, np.ones((3, 5)))
        self.assertRaises(ValueError, PSF, np.ones((4, 4)))

    def test_make_from_gaussian_noop(self):
        psf0 = PSF.from_gaussian(1e-16)
        self.assertEqual(psf0.width, 1)
        self.assertEqual(psf0.radius, 0)
        self.assertEqual(psf0.shape, (1, 1))
        self.assertTrue(np.allclose(psf0.kernel, 1.0))

    def test_make_invalid(self):
        # Raise an error if creating a PSF with a negative stdev.
        self.assertRaises(ValueError, PSF.from_gaussian, -1.0)

    def test_make_from_gaussian(self):
        psf1 = PSF.from_gaussian(1.0)
        self.assertEqual(psf1.width, 7)
        self.assertEqual(psf1.radius, 3)
        self.assertEqual(psf1.shape, (7, 7))
        self.assertAlmostEqual(np.sum(psf1.kernel), 1.0)
        self.assertAlmostEqual(psf1.kernel[3, 3], np.max(np.max(psf1.kernel[3, 3])))

    def test_convolve_identity(self):
        id_psf = PSF.from_gaussian(1e-16)

        width = 5
        height = 6
        image = np.arange(0, width * height, dtype=np.single).reshape(height, width)

        result = id_psf.convolve_image(image)
        self.assertTrue(np.allclose(result, image))

    def test_convolve_custom(self):
        # Create a non-symmetric Kernel.
        kernel = np.array([[0.0, 0.1, 0.0], [0.0, 0.8, 0.1], [0.0, 0.0, 0.0]])
        simple_psf = PSF(kernel)

        width = 5
        height = 4
        image = np.arange(0, width * height, dtype=np.single).reshape(height, width)

        result = simple_psf.convolve_image(image)
        expected = np.array(
            [
                [0.5, 1.4, 2.4, 3.4, 4.4],
                [5.0, 6.4, 7.4, 8.4, 9.4],
                [9.5, 11.4, 12.4, 13.4, 14.4],
                [12.0, 14.3, 15.2, 16.1, 17.0],
            ]
        )
        self.assertTrue(np.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
