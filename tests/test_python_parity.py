"""This is a temporary set of tests to confirm that the Python KBMOD
code is performing the same as the C++ KBMOD code.
"""

import unittest

import numpy as np

from kbmod.core.psf import convolve_psf_and_image, PSF
from kbmod.search import LayeredImage, RawImage
from kbmod.core.shift_and_stack import generate_psi_phi_images


class test_python_parity(unittest.TestCase):
    def test_convolve(self):
        height = 40
        width = 30
        img_p = (0.1 * np.arange(height * width)).reshape((height, width)).astype(np.single)

        # Mask out a few pixels.
        for py, px in [(3, 1), (10, 10), (10, 11), (10, 12), (15, 4)]:
            img_p[py, px] = np.nan

        # Create a C++ version of the image.
        img_c = RawImage(img_p)

        # Create the PSF.
        psf = PSF.make_gaussian_kernel(1.2)

        # Do the convolution with the C++ and Python functions.
        img_c.convolve(psf)
        img_p = convolve_psf_and_image(img_p, psf)

        for y in range(height):
            for x in range(width):
                if np.isnan(img_c.image[y, x]):
                    self.assertTrue(np.isnan(img_p[y, x]))
                else:
                    self.assertAlmostEqual(img_c.image[y, x], img_p[y, x], places=4)

    def test_convolve_non_unit(self):
        """Test that convolution produces the same result when the kernel does
        not sum up to 1.0."""
        height = 40
        width = 30
        img_p = (0.1 * np.arange(height * width)).reshape((height, width)).astype(np.single)

        # Mask out a few pixels.
        for py, px in [(3, 1), (10, 10), (10, 11), (10, 12), (15, 4)]:
            img_p[py, px] = np.nan

        # Create a C++ version of the image.
        img_c = RawImage(img_p)

        # Create the PSF from a Gaussian kernel and then square its values.
        psf = PSF.make_gaussian_kernel(0.9) ** 2

        # Do the convolution with the C++ and Python functions.
        img_c.convolve(psf)
        img_p = convolve_psf_and_image(img_p, psf)

        for y in range(height):
            for x in range(width):
                if np.isnan(img_c.image[y, x]):
                    self.assertTrue(np.isnan(img_p[y, x]))
                else:
                    self.assertAlmostEqual(img_c.image[y, x], img_p[y, x], places=4)

    def test_psi_phi_generation(self):
        height = 40
        width = 35
        sci = np.array([np.arange(width) for _ in range(height)], dtype=np.single)
        var = np.array([0.1 * (h + 1) * np.ones(width) for h in range(height)], dtype=np.single)
        msk = np.zeros_like(sci)

        # Mask out a few pixels.
        for py, px in [(3, 1), (10, 10), (10, 11), (10, 12), (15, 4), (35, 20), (35, 21), (35, 22)]:
            sci[py, px] = np.nan
            var[py, px] = np.nan

        # Create the PSF.
        psf = PSF.make_gaussian_kernel(1.2)

        # Create a C++ version of the image and use it to generate psi and phi images.
        img_c = LayeredImage(sci, var, msk, psf, 1.0)
        psi_c = img_c.generate_psi_image()
        phi_c = img_c.generate_phi_image()

        # Generate psi and phi via python and compare the results.
        psi_p, phi_p = generate_psi_phi_images(sci, var, psf)

        self.assertTrue(np.allclose(psi_c, psi_p, rtol=0.001, atol=0.001, equal_nan=True))
        self.assertTrue(np.allclose(phi_c, phi_p, rtol=0.001, atol=0.001, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
