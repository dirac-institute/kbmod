import unittest
from kbmod import *

class test_search(unittest.TestCase):

   def setUp(self):
      # test pass thresholds
      self.pixel_error = 1
      self.velocity_error = 0.04
      self.flux_error = 0.05

      # image properties
      self.imCount = 15
      self.dim_x = 80
      self.dim_y = 60
      self.noise_level = 1.0
      self.variance = 5.0
      self.p = psf(1.0)
      # object properties
      self.object_flux = 20.0
      self.start_x = 17
      self.start_y = 12
      self.x_vel = 21.0
      self.y_vel = 16.0
      # search parameters
      self.angle_steps = 35
      self.velocity_steps = 30
      self.min_angle = 0.0
      self.max_angle = 1.5
      self.min_vel = 5.0
      self.max_vel = 40.0

      # setup
      self.imlist = []
      for i in range(self.imCount):
         time = i/10
         im = layered_image(str(i), self.dim_x, self.dim_y, 
                 self.noise_level, self.variance, time) 
         im.add_object( self.start_x + time*self.x_vel, 
                 self.start_y + time*self.y_vel, 
                 self.object_flux, self.p)
         self.imlist.append(im)
      self.stack = image_stack(self.imlist)
      self.search = stack_search(self.stack, self.p)
      self.search.gpu( self.angle_steps, self.velocity_steps, 
                       self.min_angle, self.max_angle, self.min_vel, self.max_vel)
      

   def test_results(self):
      #self.search.save_results("./test.txt", 1)
      results = self.search.get_results(0,10)
      best = results[0]
      self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
      self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
      self.assertAlmostEqual(best.x_v/self.x_vel, 1, delta=self.velocity_error)
      self.assertAlmostEqual(best.y_v/self.y_vel, 1, delta=self.velocity_error)
      self.assertAlmostEqual(best.flux/self.object_flux, 1, delta=self.flux_error)
      

if __name__ == '__main__':
   unittest.main()

