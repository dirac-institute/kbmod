from kbmod import *
from pathlib import Path
import unittest
   
class test_layered_image(unittest.TestCase):

   def setUp(self):
       self.p = psf(1.0)

       # Create a fake layered image to use.
       self.image = layered_image("layered_test",
                                  80,    # dim_x = 80 pixels,
                                  60,    # dim_y = 60 pixels,
                                  2.0,   # noise_level
                                  4.0,   # variance
                                  10.0)  # time = 10.0


   def test_create(self):
       self.assertIsNotNone(self.image)
       self.assertEqual(self.image.get_width(), 80)
       self.assertEqual(self.image.get_height(), 60)
       self.assertEqual(self.image.get_ppi(), 80 * 60)
       self.assertEqual(self.image.get_time(), 10.0)
       self.assertEqual(self.image.get_name(), "layered_test")

       # Create a fake layered_image.
       science = self.image.get_science()
       variance = self.image.get_variance()
       mask = self.image.get_mask()
       for y in range(self.image.get_height()):
          for x in range(self.image.get_width()):
             self.assertEqual(mask.get_pixel(x, y), 0)
             self.assertEqual(variance.get_pixel(x, y), 4.0)

             # These will be potentially flakey due to the random
             # creation (but with very low probability).
             self.assertGreaterEqual(science.get_pixel(x, y), -100.0)
             self.assertLessEqual(science.get_pixel(x, y), 100.0)

             
   def test_add_object(self):
      science = self.image.get_science()
      science_50_50 = science.get_pixel(50, 50)
      self.image.add_object(50, 50, 500.0, self.p)

      science = self.image.get_science()
      self.assertLess(science_50_50, science.get_pixel(50, 50))

      
   def test_mask_threshold(self):
      masked_pixels = {}
      threshold = 20.0

      # Add an object brighter than the threshold.
      self.image.add_object(50, 50, 500.0, self.p)

      # Find all the pixels that should be masked.
      science = self.image.get_science()
      for y in range(self.image.get_height()):
         for x in range(self.image.get_width()):
            value = science.get_pixel(x, y)
            if value > threshold:
               index = self.image.get_width() * y + x
               masked_pixels[index] = True

      # Do the masking and confirm we have masked
      # at least 1 pixel.
      self.image.apply_mask_threshold(threshold)
      self.assertGreater(len(masked_pixels), 0)

      # Check that we masked the correct pixels.
      science = self.image.get_science()
      for y in range(self.image.get_height()):
         for x in range(self.image.get_width()):
            index = self.image.get_width() * y + x
            if index in masked_pixels:
               self.assertFalse(science.pixel_has_data(x, y))
            else:
               self.assertTrue(science.pixel_has_data(x, y))

               
   def test_apply_mask(self):
      # Nothing is initially masked.
      science = self.image.get_science()
      for y in range(self.image.get_height()):
         for x in range(self.image.get_width()):
            self.assertTrue(science.pixel_has_data(x, y))

      # Mask out three pixels.
      mask = self.image.get_mask()
      mask.set_pixel(10, 11, 1)
      mask.set_pixel(10, 12, 2)
      mask.set_pixel(10, 13, 3)
      self.image.set_mask(mask)

      # Apply the mask flags to only (10, 11) and (10, 13)
      self.image.apply_mask_flags(1, [])

      science = self.image.get_science()
      for y in range(self.image.get_height()):
         for x in range(self.image.get_width()):
            if x == 10 and (y == 11 or y == 13):
               self.assertFalse(science.pixel_has_data(x, y))
            else:
               self.assertTrue(science.pixel_has_data(x, y))


   def test_apply_mask_exceptions(self):
      mask = self.image.get_mask()
      mask.set_pixel(10, 11, 1)
      mask.set_pixel(10, 12, 2)
      mask.set_pixel(10, 13, 3)
      self.image.set_mask(mask)

      # Apply the mask flags to only (10, 11).
      self.image.apply_mask_flags(1, [1])

      science = self.image.get_science()
      for y in range(self.image.get_height()):
         for x in range(self.image.get_width()):
            if x == 10 and y == 13:
               self.assertFalse(science.pixel_has_data(x, y))
            else:
               self.assertTrue(science.pixel_has_data(x, y))


   def test_mask_object(self):
      # Mask a fake object at (20, 20)
      self.image.mask_object(20, 20, self.p)

      # Check that science data is masked out around (20, 20)
      # but neither the mask layer nor variance layer are changed.
      science = self.image.get_science()
      variance = self.image.get_variance()
      mask = self.image.get_mask()
      radius = self.p.get_radius()
      x_start = 20 - radius - 1
      y_start = 20 - radius - 1
      x_end = 20 + radius
      y_end = 20 + radius
      
      for y in range(self.image.get_height()):
         for x in range(self.image.get_width()):
            self.assertEqual(mask.get_pixel(x, y), 0)
            self.assertTrue(variance.pixel_has_data(x, y))

            if (x >= x_start and x <= x_end and
                y >= y_start and y <= y_end):
               self.assertFalse(science.pixel_has_data(x, y))
            else:
               self.assertTrue(science.pixel_has_data(x, y))
               
if __name__ == '__main__':
   unittest.main()

