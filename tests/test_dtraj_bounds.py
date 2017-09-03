import unittest
from kbmodpy import kbmod as kb

class test_dtraj_bounds(unittest.TestCase):

   def setUp(self):
      p = kb.psf(1.0)
      img = kb.layered_image("test", 4, 4, 0.0, 0.0, 0.0)
      stack = kb.image_stack([img])
      self.search = kb.stack_search(stack, p)

   def test_biggest_fit(self):
      self.assertEqual(self.search.biggest_fit(0,0,10,10,32), 8)
      self.assertEqual(self.search.biggest_fit(1,0,10,10,32), 1)
      self.assertEqual(self.search.biggest_fit(0,1,10,10,32), 1)
      self.assertEqual(self.search.biggest_fit(2,0,10,10,32), 2)
      self.assertEqual(self.search.biggest_fit(0,4,10,10,32), 4)
      self.assertEqual(self.search.biggest_fit(4,4,10,10,32), 4)
      self.assertEqual(self.search.biggest_fit(10,10,50,50,128), 2)
      self.assertEqual(self.search.biggest_fit(10,11,64,64,128), 1)
      self.assertEqual(self.search.biggest_fit(16,16,64,64,128), 16)
      self.assertEqual(self.search.biggest_fit(16,20,64,64,128), 4)
      self.assertEqual(self.search.biggest_fit(-150,-100,100,125,1024), 2)
      self.assertEqual(self.search.biggest_fit(-64,32,2000,2000,2048), 32)
      self.assertEqual(self.search.biggest_fit(8,8,12,17,128), 4)
      self.assertEqual(self.search.biggest_fit(8,8,11,16,128), 2)
      self.assertEqual(self.search.biggest_fit(-128,192,512,1024,512), 64)

   def test_square_sdf(self):
      self.assertEqual(self.search.square_sdf(1.0, 1.5, 1.5, 1.5, 0.5), 0.5)
      self.assertEqual(self.search.square_sdf(30, 15, 15, 50, 15), 20)
      self.assertEqual(self.search.square_sdf(2.0, 4.5, 2.5, 4.5, 1.5), 0.0)
      self.assertLess(self.search.square_sdf(10.0, 10.5, 11.5, 12.5, 11.5), 0.0)
      self.assertEqual(self.search.square_sdf(1.0, 0.0, 0.0, 0.5, 0.5), 0.0)
      self.assertEqual(self.search.square_sdf(2.0, 6.0, 6.0, 6.0, 6.0), -1.0)
      self.assertEqual(self.search.square_sdf(1.0, 1.5, 1.5, 1.5, 0.5), 0.5)

   def test_filter_bounds(self):
      t_list = []
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 50, 50, 100, 100, 0
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 50, 50, 115, 112, 0
      t_list.append(t)     
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = -40, 25, 10, 55, 0
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 280, 130, 330, 180, 0
      t_list.append(t)

      # filter trajectories out of search bounds
      t_f = self.search.filter_bounds(t_list, 10.0, 10.0, 5.0, 10.0)
      self.assertEqual(len(t_f), 2)
      self.assertEqual(t_f[0].ix, 50)
      self.assertEqual(t_f[0].iy, 50)
      self.assertEqual(t_f[0].fx, 100)
      self.assertEqual(t_f[0].fy, 100)
      self.assertEqual(t_f[1].ix, 280)
      self.assertEqual(t_f[1].iy, 130)
      self.assertEqual(t_f[1].fx, 330)
      self.assertEqual(t_f[1].fy, 180)
     
   def test_filter_bounds_depth(self):
      t_list = []
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 2, 4, 3, 5, 6
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 0, 0, 0, 0, 8
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 1, 1, -1, -1, 7
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = -40, 125, 20, 194, 0
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 280, 130, 310, 160, 1
      t_list.append(t)
      t = kb.dtraj()
      t.ix, t.iy, t.fx, t.fy, t.depth = 280, 130, 300, 170, 1
      t_list.append(t)

      # filter trajectories out of search bounds
      t_f = self.search.filter_bounds(t_list, 32.0, 32.0, 2.0, 16.0)
      self.assertEqual(len(t_f), 4)
      self.assertEqual(t_f[0].ix, 2)
      self.assertEqual(t_f[0].iy, 4)
      self.assertEqual(t_f[0].fx, 3)
      self.assertEqual(t_f[0].fy, 5)
      self.assertEqual(t_f[1].ix, 0)
      self.assertEqual(t_f[1].iy, 0)
      self.assertEqual(t_f[1].fx, 0)
      self.assertEqual(t_f[1].fy, 0)
      self.assertEqual(t_f[2].ix, -40)
      self.assertEqual(t_f[2].iy, 125)
      self.assertEqual(t_f[2].fx, 20)
      self.assertEqual(t_f[2].fy, 194)
      self.assertEqual(t_f[3].ix, 280)
      self.assertEqual(t_f[3].iy, 130)
      self.assertEqual(t_f[3].fx, 310)
      self.assertEqual(t_f[3].fy, 160)

if __name__ == '__main__':
   unittest.main()
