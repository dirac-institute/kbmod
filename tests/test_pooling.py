import unittest
from kbmodpy import kbmod as kb
import numpy as np

class test_pooling(unittest.TestCase):

   def setUp(self):
      pass

   def test_pooling(self):
      depth = 10
      res = 2**depth
      im = kb.layered_image("test", res, res, 0.0, 1.0, 0.0)
      im = im.get_science()
      for _ in range(depth):
         im = im.pool(1)
      im = np.array(im)
      self.assertEqual(im[0][0], 0.0)

      im = kb.layered_image("test", res, res, 0.0, 1.0, 0.0)
      im = im.get_science()
      for _ in range(depth):
         im = im.pool(0)
      im = np.array(im)
      self.assertEqual(im[0][0], 0.0)

      im = kb.layered_image("test", res, res, 3.0, 9.0, 0.0)
      im = im.get_science()
      test_high = 142.6
      test_low = -302.2
      im.set_pixel(51,55, test_high)
      im.set_pixel(20,18, test_low)
      # reduce to max
      imax = im.pool(1)
      for _ in range(depth-1):
         imax = imax.pool(1)
      imax = np.array(imax)
      self.assertAlmostEqual(imax[0][0], test_high, delta=0.001)

      #reduce to min
      imin = im.pool(0)
      for _ in range(depth):
         imin = imin.pool(0)
      imin = np.array(imin)
      self.assertAlmostEqual(imin[0][0], test_low, delta=0.001)
     
   def test_all_pix(self):
      im = kb.raw_image(100,100)
      test_val = 402.0
      im.set_all(test_val)
      for _ in range(8):
         im = im.pool(1)
      self.assertAlmostEqual(np.array(im)[0][0], test_val, delta=0.001)
 
if __name__ == '__main__':
   unittest.main()
