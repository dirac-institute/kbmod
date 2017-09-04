import unittest
from kbmodpy import kbmod as kb

class test_multires(unittest.TestCase):

   def setUp(self):
      # test pass thresholds
      im_count = 10
      self.ix = 136
      self.iy = 103
      self.xv = 34.0
      self.yv = 21.0
      self.flux = 350.0
      p = kb.psf(1.0)

      imgs = []
      for i in range(im_count):
         im = kb.layered_image("im"+str(i+1), 
            500, 500, 0.0, 100.0, i*0.1)
         im.add_object(self.ix+0.1*i*self.xv, 
                       self.iy+0.1*i*self.yv, 
                       self.flux, p)
         imgs.append(im)
      stack = kb.image_stack(imgs)
      self.search = kb.stack_search(stack, p)

   def test_extreme_in_region(self):
      results = self.search.region_search(self.xv, self.yv, 
         10.0, 12.0, 3)
      r = results[0]
      self.assertEqual(r.ix,136)
      self.assertEqual(r.iy,103)
      self.assertAlmostEqual(r.flux, self.flux, delta=60)
      #self.assertEqual(r.

if __name__ == '__main__':
   unittest.main()
