import unittest
from kbmod import psf

class test_psf(unittest.TestCase):

   def setUp(self):
      self.psf_count = 12
      sigma_list = range(self.psf_count)
      self.psf_list = [psf(x/5+0.2) for x in sigma_list]

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

if __name__ == '__main__':
   unittest.main()
