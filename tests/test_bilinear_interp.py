import unittest
import numpy
from kbmodpy import kbmod as kb

class test_bilinear_interp(unittest.TestCase):

   def setUp(self):
      self.im_count = 5
      p = kb.psf(0.05)
      self.images = []
      for c in range(self.im_count):
         im = kb.layered_image(str(c), 10, 10, 0.0, 1.0, c)
         im.add_object( 2+c*0.5+0.5, 2+c*0.5+0.5, 1, p)
         self.images.append(im)
      

   def test_pixels(self):
 
      d = 0.001

      pixels = self.images[0].science()
      self.assertAlmostEqual(pixels.item(2,2), 1, delta=d)
      self.assertAlmostEqual(pixels.item(3,2), 0, delta=d)
      self.assertAlmostEqual(pixels.item(2,3), 0, delta=d)
      self.assertAlmostEqual(pixels.item(1,2), 0, delta=d)
      self.assertAlmostEqual(pixels.item(2,1), 0, delta=d)

      pixels = self.images[1].science()
      self.assertAlmostEqual(pixels.item(2,2), 0.25, delta=d)
      self.assertAlmostEqual(pixels.item(3,2), 0.25, delta=d)
      self.assertAlmostEqual(pixels.item(2,3), 0.25, delta=d)
      self.assertAlmostEqual(pixels.item(3,3), 0.25, delta=d)
      self.assertAlmostEqual(pixels.item(2,1), 0, delta=d)

if __name__ == '__main__':
   unittest.main()

