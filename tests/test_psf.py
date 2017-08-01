import unittest
from kbmod import psf

class test_psf(unittest.TestCase):

   def setUp(self):
      self.psf_count = 12
      sigma_list = range(self.psf_count)
      self.psf_list = [psf(x/5+0.2) for x in sigma_list]

   def test_sum(self):
      for p in self.psf_list:
         self.assertGreater( p.get_sum(), 0.95 )

   def test_square(self):
      for p in self.psf_list:
         x = p.get_sum()
         p.square_psf()
         self.assertGreater( x, p.get_sum())

if __name__ == '__main__':
   unittest.main()
