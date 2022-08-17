import unittest
from kbmod import *

class test_predicted_position(unittest.TestCase):

    def setUp(self):
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

        # create a trajectory for the object
        self.trj = trajectory()
        self.trj.x = self.start_x
        self.trj.y = self.start_y
        self.trj.x_v = self.x_vel
        self.trj.y_v = self.y_vel

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

    def test_single_prediction(self):
        p = self.search.get_traj_pos(self.trj, 0)
        self.assertAlmostEqual(p.x, self.trj.x, delta = 1e-8)
        self.assertAlmostEqual(p.y, self.trj.y, delta = 1e-8)
          
        p = self.search.get_traj_pos(self.trj, 10)
        self.assertAlmostEqual(p.x, self.trj.x + 0.5 * self.trj.x_v, 
                               delta=1e-8)
        self.assertAlmostEqual(p.y, self.trj.y + 0.5 * self.trj.y_v, 
                               delta=1e-8)
            
    def test_all_predictions(self):
        p_arr = self.search.get_traj_positions(self.trj)
        self.assertEqual(len(p_arr), self.imCount)
        
        for i in range(self.imCount):
            t = 0.05 * float(i)
            self.assertAlmostEqual(p_arr[i].x, self.trj.x + t * self.trj.x_v,
                                   delta=1e-5)
            self.assertAlmostEqual(p_arr[i].y, self.trj.y + t * self.trj.y_v,
                                   delta=1e-5)

if __name__ == '__main__':
   unittest.main()

