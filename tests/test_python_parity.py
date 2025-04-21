"""This is a temporary set of tests to confirm that the Python KBMOD
code is performing the same as the C++ KBMOD code.
"""

import unittest

import numpy as np

from kbmod.core.psf import convolve_psf_and_image, PSF
from kbmod.search import RawImage


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


if __name__ == "__main__":
    unittest.main()
