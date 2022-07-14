import unittest
from kbmod import *

class test_search(unittest.TestCase):

   def setUp(self):
      # test pass thresholds
      self.pixel_error = 0
      self.velocity_error = 0.05
      self.flux_error = 0.15

      # image properties
      self.imCount = 20
      self.dim_x = 80
      self.dim_y = 60
      self.noise_level = 8.0
      self.variance = self.noise_level**2
      self.p = psf(1.0)

      # object properties
      self.object_flux = 250.0
      self.start_x = 17
      self.start_y = 12
      self.x_vel = 21.0
      self.y_vel = 16.0

      # search parameters
      self.angle_steps = 150
      self.velocity_steps = 150
      self.min_angle = 0.0
      self.max_angle = 1.5
      self.min_vel = 5.0
      self.max_vel = 40.0

      # create image set with single moving object
      self.imlist = []
      for i in range(self.imCount):
         time = i/self.imCount
         im = layered_image(str(i), self.dim_x, self.dim_y, 
                            self.noise_level, self.variance, time)
         im.add_object(self.start_x + time*self.x_vel+0.5,
                       self.start_y + time*self.y_vel+0.5,
                       self.object_flux, self.p)
         self.imlist.append(im)
      self.stack = image_stack(self.imlist)
      self.search = stack_search(self.stack, self.p)
      self.search.gpu(self.angle_steps, self.velocity_steps,
                      self.min_angle, self.max_angle, self.min_vel,
                      self.max_vel, int(self.imCount/2))

   def test_psiphi(self):
      p = psf(0.00001)

      # Image1 has a single object.
      image1 = layered_image("test1", 5, 10, 2.0, 4.0, 1.0)
      image1.add_object(3.5, 2.5, 400.0, p)

      # Image2 has a single object and a masked pixel.
      image2 = layered_image("test2", 5, 10, 2.0, 4.0, 2.0)
      image2.add_object(2.5, 4.5, 400.0, p)
      mask = image2.get_mask()
      mask.set_pixel(4, 9, 1)
      image2.set_mask(mask)
      image2.apply_mask_flags(1, [])

      # Create a stack from the two objects.
      stack = image_stack([image1, image2])
      search = stack_search(stack, p)

      # Generate psi and phi.
      search.prepare_psi_phi()
      psi = search.get_psi_images()
      phi = search.get_phi_images()

      # Test phi and psi for image1.
      sci = image1.get_science()
      var = image1.get_variance()
      for x in range(5):
         for y in range(10):
            self.assertAlmostEqual(psi[0].get_pixel(x, y),
                                   sci.get_pixel(x, y)/var.get_pixel(x, y),
                                   delta = 1e-6)
            self.assertAlmostEqual(phi[0].get_pixel(x, y),
                                   1.0 / var.get_pixel(x, y),
                                   delta = 1e-6)

      # Test phi and psi for image2.
      sci = image2.get_science()
      var = image2.get_variance()
      for x in range(5):
         for y in range(10):
            if x == 4 and y == 9:
               self.assertFalse(psi[1].pixel_has_data(x, y))
               self.assertFalse(phi[1].pixel_has_data(x, y))
            else:
               self.assertAlmostEqual(psi[1].get_pixel(x, y),
                                      sci.get_pixel(x, y)/var.get_pixel(x, y),
                                      delta = 1e-6)
               self.assertAlmostEqual(phi[1].get_pixel(x, y),
                                      1.0 / var.get_pixel(x, y),
                                      delta = 1e-6)

   def test_results(self):
      results = self.search.get_results(0,10)
      best = results[0]
      self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
      self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
      self.assertAlmostEqual(best.x_v/self.x_vel, 1, delta=self.velocity_error)
      self.assertAlmostEqual(best.y_v/self.y_vel, 1, delta=self.velocity_error)
      self.assertAlmostEqual(best.flux/self.object_flux, 1, delta=self.flux_error)
      

if __name__ == '__main__':
   unittest.main()

