import unittest

from kbmod.evaluate import *
from kbmod.search import *


def make_trajectory(x, y, xv, yv):
    """
    Make a fake trajectory with the given parameters.

    Arguments:
        x : int
            The starting x coordinate in pixels
        y : int
            The starting y coordinate in pixels
        xv : float
            The x velocity in pixels per day
        yv : float
            The y velocity in pixels per day

    Returns:
        A trajectory object.
    """
    t = trajectory()
    t.x = x
    t.y = y
    t.x_v = xv
    t.y_v = yv
    return t


class test_evaluate(unittest.TestCase):
    def test_ave_distances(self):
        ave_dist = ave_trajectory_distance(
            make_trajectory(5, 6, 10.0, -1.0), make_trajectory(5, 7, -10.0, 2.0), times=[0.0, 1.0]
        )
        self.assertAlmostEqual(ave_dist, 10.698039027)

    def test_match_on_start(self):
        trjA = [
            make_trajectory(5, 5, 1.0, -1.0),
            make_trajectory(10, 5, 1.0, -1.0),
            make_trajectory(5, 20, 1.0, -1.0),
        ]
        trjB = [
            make_trajectory(5, 5, 1.0, -1.0),
            make_trajectory(10, 6, 1.0, -1.0),
            make_trajectory(5, 200, 1.0, -1.0),
        ]

        match = find_unique_overlap(trjA, trjB, 2.0, [0.0])
        self.assertEqual(len(match), 2)
        self.assertEqual(match[0].x, 5)
        self.assertEqual(match[0].y, 5)
        self.assertEqual(match[1].x, 10)
        self.assertEqual(match[1].y, 5)

        diff = find_set_difference(trjA, trjB, 2.0, [0.0])
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0].x, 5)
        self.assertEqual(diff[0].y, 20)

    def test_match_on_end(self):
        trjA = [
            make_trajectory(5, 5, 2.0, 0.0),
            make_trajectory(10, 5, 1.0, -1.0),
            make_trajectory(7, 20, 1.0, 0.0),
        ]
        trjB = [
            make_trajectory(5, 5, 0.0, -2.0),
            make_trajectory(10, 6, 1.0, -1.5),
            make_trajectory(7, 16, 1.0, 2.0),
        ]

        match = find_unique_overlap(trjA, trjB, 2.0, [2.0])
        self.assertEqual(len(match), 2)
        self.assertEqual(match[0].x, 10)
        self.assertEqual(match[0].y, 5)
        self.assertEqual(match[1].x, 7)
        self.assertEqual(match[1].y, 20)

        diff = find_set_difference(trjA, trjB, 2.0, [2.0])
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0].x, 5)
        self.assertEqual(diff[0].y, 5)


if __name__ == "__main__":
    unittest.main()
