import unittest
from kbmodpy import kbmod as kb

class test_dtraj_regions(unittest.TestCase):

   def setUp(self):
      # test pass thresholds
      im = kb.layered_image("",171,111,5.0,25.0,0)
      stack = kb.image_stack([im])
      p = kb.psf(1.0)
      self.search = kb.stack_search(stack, p)

      max_img = im.get_science()
      max_img.set_pixel(38,39,117)
      max_img.set_pixel(34,53,1000)
      max_img.set_pixel(50,37, 1000)
      max_img.set_pixel(70,72, 1000)
      self.pooled_max = []
      while (max_img.get_ppi() > 1):
         self.pooled_max.append(max_img)
         max_img = max_img.pool_max()

      min_img = im.get_science()
      self.pooled_min = []
      while (min_img.get_ppi() > 1):
         self.pooled_min.append(min_img)
         min_img = min_img.pool_min()

   def test_extreme_in_region(self):
      self.assertEqual(self.pooled_max[0].get_pixel(38,39), 117)
      self.assertLess(self.pooled_min[4].get_pixel(2,1), -5.0)
      self.assertEqual(self.search.extreme_in_region(1.6, 1.7, 32, 
         self.pooled_max, kb.pool_max), 117)

if __name__ == '__main__':
   unittest.main()
