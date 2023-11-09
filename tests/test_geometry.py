import numpy as np
import unittest

from kbmod.region_search import geometry


class test_geometry(unittest.TestCase):
    def test_ang2unitvec(self):
        (x, y, z) = geometry.ang2unitvec(90.0, 0.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)

        (x, y, z) = geometry.ang2unitvec(
            np.array([0.0, 90.0, 45.0, 90.0, 270.0]),
            np.array([0.0, 90.0, 0.0, 45.0, 0.0]),
        )
        self.assertTrue(np.allclose(x, [1.0, 0.0, 0.707106781, 0.0, 0.0]))
        self.assertTrue(np.allclose(y, [0.0, 0.0, 0.707106781, 0.707106781, -1.0]))
        self.assertTrue(np.allclose(z, [0.0, 1.0, 0.0, 0.707106781, 0.0]))

    def test_unitvec2ang(self):
        (r, d) = geometry.unitvec2ang(0.0, 1.0, 0.0)
        self.assertAlmostEqual(r, 90.0)
        self.assertAlmostEqual(d, 0.0)

        (r, d) = geometry.unitvec2ang(
            np.array([1.0, 0.707106781, 0.0, -1.0, 0.0]),
            np.array([0.0, 0.707106781, 0.707106781, 0.0, -1.0]),
            np.array([0.0, 0.0, 0.707106781, 0.0, 0.0]),
        )
        self.assertTrue(np.allclose(r, [0.0, 45.0, 90.0, 180.0, 270.0]))
        self.assertTrue(np.allclose(d, [0.0, 0.0, 45.0, 0.0, 0.0]))

    def test_angular_distance(self):
        pts1 = np.array(
            [
                [0.5, 1.0, 0.3],
                [0.0, 1.0, 0.0],
                [0.0, 0.707106781, 0.707106781],
                [0.0, 0.0, 1.0],
                [10.0, 7.0, 10.0],
            ]
        )
        pts2 = np.array(
            [
                [0.5, 1.0, 0.3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [5.0, 9.0, -2.0],
            ]
        )
        expected = np.array([0.0, np.pi / 2, np.pi / 4.0, np.pi, 0.9740718])

        ang_dist = geometry.angular_distance(pts1, pts2)
        self.assertTrue(np.allclose(ang_dist, expected, atol=1e-05))
