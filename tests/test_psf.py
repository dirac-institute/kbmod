import unittest

from kbmod.search import psf


class test_psf(unittest.TestCase):
    def setUp(self):
        self.psf_count = 12
        sigma_list = range(self.psf_count)
        self.psf_list = [psf(x / 5 + 0.2) for x in sigma_list]

    def test_make_and_copy(self):
        psf1 = psf(1.0)
        self.assertEqual(psf1.get_size(), 25)
        self.assertEqual(psf1.get_dim(), 5)
        self.assertEqual(psf1.get_radius(), 2)

        # Make a copy.
        psf2 = psf(psf1)
        self.assertEqual(psf2.get_size(), 25)
        self.assertEqual(psf2.get_dim(), 5)
        self.assertEqual(psf2.get_radius(), 2)

        kernel1 = psf1.get_kernel()
        kernel2 = psf2.get_kernel()
        for i in range(psf1.get_size()):
            self.assertEqual(kernel1[i], kernel2[i])

    # Test the creation of a delta function (no-op) PSF.
    def test_no_op(self):
        psf1 = psf(0.000001)
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
            x = psf(p)
            self.assertEqual(x.get_sum(), x_sum)

            # Since each pixel value is squared the sum of the
            # PSF should be smaller.
            p.square_psf()
            self.assertGreater(x.get_sum(), p.get_sum())

            # Squaring the PSF should not change any of the parameters.
            self.assertEqual(x.get_dim(), p.get_dim())
            self.assertEqual(x.get_size(), p.get_size())
            self.assertEqual(x.get_radius(), p.get_radius())


if __name__ == "__main__":
    unittest.main()
