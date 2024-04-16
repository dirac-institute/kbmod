import math
import numpy as np
import unittest

from kbmod.search import Trajectory, pixel_value_valid
from kbmod.trajectory_utils import make_trajectory


class test_common(unittest.TestCase):
    def test_pixel_value_valid(self):
        self.assertTrue(pixel_value_valid(1.0))
        self.assertTrue(pixel_value_valid(-1.0))
        self.assertTrue(pixel_value_valid(0.0))
        self.assertFalse(pixel_value_valid(math.nan))
        self.assertFalse(pixel_value_valid(np.nan))

    def test_trajectory_predict(self):
        trj = make_trajectory(x=5, y=10, vx=2.0, vy=-1.0)
        # With centered=false the trajectories start at the pixel edge.
        self.assertEqual(trj.get_x_pos(0.0, False), 5.0)
        self.assertEqual(trj.get_y_pos(0.0, False), 10.0)
        self.assertEqual(trj.get_x_pos(1.0, False), 7.0)
        self.assertEqual(trj.get_y_pos(1.0, False), 9.0)
        self.assertEqual(trj.get_x_pos(2.0, False), 9.0)
        self.assertEqual(trj.get_y_pos(2.0, False), 8.0)

        # Centering moves things by half a pixel.
        self.assertEqual(trj.get_x_pos(0.0), 5.5)
        self.assertEqual(trj.get_y_pos(0.0), 10.5)
        self.assertEqual(trj.get_x_pos(1.0), 7.5)
        self.assertEqual(trj.get_y_pos(1.0), 9.5)
        self.assertEqual(trj.get_x_pos(2.0), 9.5)
        self.assertEqual(trj.get_y_pos(2.0), 8.5)

        # Predicting the index gives a floored integer of the centered prediction.
        self.assertEqual(trj.get_x_index(0.0), 5)
        self.assertEqual(trj.get_y_index(0.0), 10)
        self.assertEqual(trj.get_x_index(1.0), 7)
        self.assertEqual(trj.get_y_index(1.0), 9)

    def test_trajectory_is_close(self):
        trj = make_trajectory(x=5, y=10, vx=2.0, vy=-1.0)

        trj2 = make_trajectory(x=5, y=10, vx=2.0, vy=-1.0)
        self.assertTrue(trj.is_close(trj2, 1e-6, 1e-6))

        trj3 = make_trajectory(x=6, y=9, vx=2.1, vy=-1.1)
        self.assertFalse(trj.is_close(trj3, 0.01, 0.01))
        self.assertFalse(trj.is_close(trj3, 1.0, 0.01))
        self.assertFalse(trj.is_close(trj3, 10.0, 0.01))
        self.assertTrue(trj.is_close(trj3, 2.0, 0.5))

        trj3 = make_trajectory(x=5, y=10, vx=2.01, vy=-0.99)
        self.assertFalse(trj.is_close(trj3, 0.0001, 0.0001))
        self.assertTrue(trj.is_close(trj3, 0.0001, 0.02))


if __name__ == "__main__":
    unittest.main()
