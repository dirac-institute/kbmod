import math
import numpy as np
import unittest
import warnings

from kbmod.search import kb_has_gpu, Trajectory, pixel_value_valid


class test_common(unittest.TestCase):
    @unittest.skipIf(kb_has_gpu(), "Skipping test (GPU detected)")
    def test_warning_no_GPU(self):
        """Throw a loud warning if you are running tests without GPU."""
        warnings.warn("\n\n*** WARNING: SKIPPING GPU TESTS ***\n\n")

    def test_pixel_value_valid(self):
        self.assertTrue(pixel_value_valid(1.0))
        self.assertTrue(pixel_value_valid(-1.0))
        self.assertTrue(pixel_value_valid(0.0))
        self.assertFalse(pixel_value_valid(math.nan))
        self.assertFalse(pixel_value_valid(np.nan))

    def test_trajectory_create(self):
        # All defaults
        trj1 = Trajectory()
        self.assertEqual(trj1.x, 0)
        self.assertEqual(trj1.y, 0)
        self.assertEqual(trj1.vx, 0.0)
        self.assertEqual(trj1.vy, 0.0)
        self.assertEqual(trj1.flux, 0.0)
        self.assertEqual(trj1.lh, 0.0)
        self.assertEqual(trj1.obs_count, 0)

        # All specified
        trj2 = Trajectory(x=1, y=2, vx=3.0, vy=4.0, flux=5.0, lh=6.0, obs_count=7)
        self.assertEqual(trj2.x, 1)
        self.assertEqual(trj2.y, 2)
        self.assertEqual(trj2.vx, 3.0)
        self.assertEqual(trj2.vy, 4.0)
        self.assertEqual(trj2.flux, 5.0)
        self.assertEqual(trj2.lh, 6.0)
        self.assertEqual(trj2.obs_count, 7)

        # Some specified, some defaults
        trj3 = Trajectory(y=2, vx=3.0, vy=-4.0, obs_count=7)
        self.assertEqual(trj3.x, 0)
        self.assertEqual(trj3.y, 2)
        self.assertEqual(trj3.vx, 3.0)
        self.assertEqual(trj3.vy, -4.0)
        self.assertEqual(trj3.flux, 0.0)
        self.assertEqual(trj3.lh, 0.0)
        self.assertEqual(trj3.obs_count, 7)

        # Four specified by order
        trj4 = Trajectory(4, 3, 2.0, 1.0)
        self.assertEqual(trj4.x, 4)
        self.assertEqual(trj4.y, 3)
        self.assertEqual(trj4.vx, 2.0)
        self.assertEqual(trj4.vy, 1.0)
        self.assertEqual(trj4.flux, 0.0)
        self.assertEqual(trj4.lh, 0.0)
        self.assertEqual(trj4.obs_count, 0)

    def test_trajectory_predict(self):
        trj = Trajectory(x=5, y=10, vx=2.0, vy=-1.0)
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


if __name__ == "__main__":
    unittest.main()
