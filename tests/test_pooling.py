import unittest
from kbmodpy import kbmod as kb
import numpy as np

class test_pooling(unittest.TestCase):

   def setUp(self):
      self.p = kb.psf(1.0)

   def test_pooling_max_small(self):
      """
      Tests max pooling on a manually constructed 4 x 4 example.
      """
      im = kb.layered_image("test", 4, 4, 0.0, 1.0, 0.0, self.p)
      sci = im.get_science()
      sci.set_all(1.0)
      sci.set_pixel(0, 0, 0.0)
      sci.set_pixel(0, 1, 2.0)
      sci.set_pixel(1, 3, 1.5)
      sci.set_pixel(3, 0, 3.0)
      sci.set_pixel(3, 1, 3.0)
      sci.set_pixel(3, 2, 0.5)
      pooled = sci.pool_max()

      self.assertEqual(pooled.get_height(), 2)
      self.assertEqual(pooled.get_width(), 2)
      self.assertEqual(pooled.get_ppi(), 4)
      self.assertAlmostEqual(pooled.get_pixel(0, 0), 2.0, delta=1e-8)
      self.assertAlmostEqual(pooled.get_pixel(0, 1), 1.5, delta=1e-8)
      self.assertAlmostEqual(pooled.get_pixel(1, 0), 3.0, delta=1e-8)
      self.assertAlmostEqual(pooled.get_pixel(1, 1), 1.0, delta=1e-8)

   def test_pooling_max_small(self):
      """
      Tests min pooling on a manually constructed 4 x 4 example.
      """
      im = kb.layered_image("test", 4, 4, 0.0, 1.0, 0.0, self.p)
      sci = im.get_science()
      sci.set_all(1.0)
      sci.set_pixel(0, 0, 0.0)
      sci.set_pixel(0, 1, 2.0)
      sci.set_pixel(1, 3, 1.5)
      sci.set_pixel(3, 0, 3.0)
      sci.set_pixel(3, 1, 3.0)
      sci.set_pixel(3, 2, 0.5)
      pooled = sci.pool_min()

      self.assertEqual(pooled.get_height(), 2)
      self.assertEqual(pooled.get_width(), 2)
      self.assertEqual(pooled.get_ppi(), 4)
      self.assertAlmostEqual(pooled.get_pixel(0, 0), 0.0, delta=1e-8)
      self.assertAlmostEqual(pooled.get_pixel(0, 1), 1.0, delta=1e-8)
      self.assertAlmostEqual(pooled.get_pixel(1, 0), 1.0, delta=1e-8)
      self.assertAlmostEqual(pooled.get_pixel(1, 1), 0.5, delta=1e-8)

   def test_pooling_to_one(self):
      depth = 10
      res = 2**depth
      im = kb.layered_image("test", res, res, 0.0, 1.0, 0.0, self.p)
      im = im.get_science()
      for _ in range(depth):
         im = im.pool_max()
      im = np.array(im)
      self.assertEqual(im[0][0], 0.0)

      im = kb.layered_image("test", res, res, 0.0, 1.0, 0.0, self.p)
      im = im.get_science()
      for _ in range(depth):
         im = im.pool_min()
      im = np.array(im)
      self.assertEqual(im[0][0], 0.0)

      im = kb.layered_image("test", res, res, 3.0, 9.0, 0.0, self.p)
      im = im.get_science()
      test_high = 142.6
      test_low = -302.2
      im.set_pixel(51, 55, test_high)
      im.set_pixel(20, 18, test_low)
      # reduce to max
      imax = im.pool_max()
      for _ in range(depth-1):
         imax = imax.pool_max()
      imax = np.array(imax)
      self.assertAlmostEqual(imax[0][0], test_high, delta=0.001)

      #reduce to min
      imin = im.pool_min()
      for _ in range(depth):
         imin = imin.pool_min()
      imin = np.array(imin)
      self.assertAlmostEqual(imin[0][0], test_low, delta=0.001)
     
   def test_all_pix(self):
      im = kb.raw_image(100,100)
      test_val = 402.0
      im.set_all(test_val)
      for _ in range(8):
         im = im.pool_max()
      self.assertAlmostEqual(np.array(im)[0][0], test_val, delta=0.001)
 
if __name__ == '__main__':
   unittest.main()
