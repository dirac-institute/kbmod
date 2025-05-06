"""This is a temporary set of tests to confirm that the Python KBMOD
code is performing the same as the C++ KBMOD code.
"""

import unittest

import numpy as np

from kbmod.core.psf import convolve_psf_and_image, PSF
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.image_utils import extract_sci_images_from_stack, extract_var_images_from_stack
from kbmod.search import (
    fill_psi_phi_array,
    fill_psi_phi_array_from_image_stack,
    LayeredImage,
    RawImage,
    PsiPhiArray,
)
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

    def test_single_psi_phi_image_generation(self):
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

    def test_psi_phi_array_generation(self):
        num_times = 10
        width = 200
        height = 300
        times = np.arange(num_times)
        fake_ds = FakeDataSet(width, height, times)

        # Create the PsiPhiArray from the ImageStack.
        arr_c = PsiPhiArray()
        fill_psi_phi_array_from_image_stack(arr_c, fake_ds.stack, 2)

        # Process the images using the Python functions.
        psi_arr = []
        phi_arr = []
        for idx in range(num_times):
            layered_img = fake_ds.stack.get_single_image(idx)
            psi, phi = generate_psi_phi_images(
                layered_img.sci,
                layered_img.var,
                layered_img.get_psf(),
            )
            psi_arr.append(psi)
            phi_arr.append(phi)

        # Create the PsiPhiArray from the Python processed data.
        arr_p = PsiPhiArray()
        fill_psi_phi_array(arr_p, 2, psi_arr, phi_arr, times)

        # Check that the arrays' metadata are the same.
        self.assertEqual(arr_c.num_times, arr_p.num_times)
        self.assertEqual(arr_c.width, arr_p.width)
        self.assertEqual(arr_c.height, arr_p.height)
        self.assertEqual(arr_c.pixels_per_image, arr_p.pixels_per_image)
        self.assertEqual(arr_c.num_entries, arr_p.num_entries)
        self.assertEqual(arr_c.total_array_size, arr_p.total_array_size)
        self.assertEqual(arr_c.block_size, arr_p.block_size)
        self.assertAlmostEqual(arr_c.psi_min_val, arr_p.psi_min_val, places=3)
        self.assertAlmostEqual(arr_c.psi_max_val, arr_p.psi_max_val, places=3)
        self.assertAlmostEqual(arr_c.phi_min_val, arr_p.phi_min_val, places=3)
        self.assertAlmostEqual(arr_c.phi_max_val, arr_p.phi_max_val, places=3)
        self.assertAlmostEqual(arr_c.psi_scale, arr_p.psi_scale, places=3)
        self.assertAlmostEqual(arr_c.phi_scale, arr_p.phi_scale, places=3)

        # Check the extracted pixels.
        for t in range(num_times):
            for y in range(height):
                for x in range(width):
                    psi_phi_c = arr_c.read_psi_phi(t, y, x)
                    psi_phi_p = arr_p.read_psi_phi(t, y, x)
                    self.assertAlmostEqual(psi_phi_c.psi, psi_phi_p.psi, places=3)
                    self.assertAlmostEqual(psi_phi_c.phi, psi_phi_p.phi, places=3)


if __name__ == "__main__":
    unittest.main()
