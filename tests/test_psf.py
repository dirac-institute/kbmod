import math
import numpy as np
import unittest

from kbmod.search import PSF


class test_PSF(unittest.TestCase):
    def setUp(self):
        self.psf_count = 12
        sigma_list = range(self.psf_count)
        self.psf_list = [PSF(x / 5 + 0.2) for x in sigma_list]

    def test_make_noop(self):
        psf0 = PSF()
        self.assertEqual(psf0.get_size(), 1)
        self.assertEqual(psf0.get_dim(), 1)
        self.assertEqual(psf0.get_radius(), 0)

        kernel0 = psf0.get_kernel()
        self.assertEqual(len(kernel0), 1)
        self.assertEqual(kernel0[0], 1.0)

    def test_make_invalid(self):
        # Raise an error if creating a PSF with a negative stdev.
        self.assertRaises(RuntimeError, PSF, -1.0)

    def test_make_from_array(self):
        arr = np.full((3, 3), 1.0 / 9.0)
        psf_arr = PSF(arr)
        self.assertEqual(psf_arr.get_size(), 9)
        self.assertEqual(psf_arr.get_dim(), 3)

        # We get an error if we include a NaN.
        arr[0][0] = math.nan
        self.assertRaises(RuntimeError, PSF, arr)

        # We get an error if we include a inf.
        arr[0][0] = math.inf
        self.assertRaises(RuntimeError, PSF, arr)

    def test_to_string(self):
        result = self.psf_list[0].__str__()
        self.assertGreater(len(result), 1)

    def test_make_and_copy(self):
        psf1 = PSF(1.0)
        self.assertEqual(psf1.get_size(), 25)
        self.assertEqual(psf1.get_dim(), 5)
        self.assertEqual(psf1.get_radius(), 2)

        # Make a copy.
        psf2 = PSF(psf1)
        self.assertEqual(psf2.get_size(), 25)
        self.assertEqual(psf2.get_dim(), 5)
        self.assertEqual(psf2.get_radius(), 2)

        kernel1 = psf1.get_kernel()
        kernel2 = psf2.get_kernel()
        for i in range(psf1.get_size()):
            self.assertEqual(kernel1[i], kernel2[i])

    # Test the creation of a delta function (no-op) PSF.
    def test_no_op(self):
        psf1 = PSF(0.000001)
        self.assertEqual(psf1.get_size(), 1)
        self.assertEqual(psf1.get_dim(), 1)
        self.assertEqual(psf1.get_radius(), 0)
        kernel = psf1.get_kernel()
        self.assertAlmostEqual(kernel[0], 1.0, delta=0.001)

    # Test that the PSF sums to close to 1.
    def test_sum(self):
        for p in self.psf_list:
            self.assertGreater(p.get_sum(), 0.95)

    def test_square(self):
        for p in self.psf_list:
            x_sum = p.get_sum()

            # Make a copy and confirm that the copy preserves the sum.
            x = PSF(p)
            self.assertEqual(x.get_sum(), x_sum)

            # Since each pixel value is squared the sum of the
            # PSF should be smaller.
            p.square_psf()
            self.assertGreater(x.get_sum(), p.get_sum())

            # Squaring the PSF should not change any of the parameters.
            self.assertEqual(x.get_dim(), p.get_dim())
            self.assertEqual(x.get_size(), p.get_size())
            self.assertEqual(x.get_radius(), p.get_radius())

    def test_stats_string(self):
        pfs_gauss = PSF(0.5)
        gauss_str = pfs_gauss.stats_string()
        self.assertTrue("Gaussian" in gauss_str)
        self.assertTrue("radius = 1" in gauss_str)

        psf_arr = PSF(np.full((3, 3), 1.0 / 9.0))
        arr_str = psf_arr.stats_string()
        self.assertTrue("Manual" in arr_str)
        self.assertTrue("radius = 1" in arr_str)

    def test_is_close(self):
        pfs_gauss1 = PSF(0.5)
        pfs_gauss2 = PSF(0.50000001)
        pfs_gauss3 = PSF(0.51)
        self.assertTrue(pfs_gauss1.is_close(pfs_gauss2, 0.01))
        self.assertTrue(pfs_gauss2.is_close(pfs_gauss2, 0.01))
        self.assertFalse(pfs_gauss1.is_close(pfs_gauss3, 0.01))
        self.assertFalse(pfs_gauss3.is_close(pfs_gauss1, 0.01))

        psf_noop = PSF()
        self.assertFalse(pfs_gauss1.is_close(psf_noop, 0.01))


if __name__ == "__main__":
    unittest.main()
