import unittest
from kbmodpy import kbmod as kb

class test_dtraj_regions(unittest.TestCase):

   def setUp(self):
      # test pass thresholds
      
      max_img = kb.raw_image(170, 101)
      self.pooled_max = []
      while (max_img.get_ppi() > 1):
         self.pooled_max.append(max_img)
         max_img = max_img.pool_max()
      

