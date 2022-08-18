import unittest
from astropy.coordinates import SkyCoord
from kbmod import *
from known_objects import *

class test_known_objects(unittest.TestCase):

    def _make_pos_from_trajectories(self, trjs):
        result = []

        for i in range(len(self.trjs)):
            pos = [None] * self.num_time_steps
            for t in range(self.num_time_steps):
                x_t = self.trjs[i].x + float(t) * self.trjs[i].x_v
                y_t = self.trjs[i].y + float(t) * self.trjs[i].y_v
                pos[t] = SkyCoord(x_t, y_t, unit="deg")
            result.append(pos)
        return result

    def setUp(self):
        self.num_time_steps = 10

        # Create trajectories for the fake objects
        self.trjs = []

        trj0 = trajectory()
        trj0.x = 50
        trj0.y = 60
        trj0.x_v = 0.01
        trj0.y_v = -0.02
        self.trjs.append(trj0)

        trj1 = trajectory()
        trj1.x = 30
        trj1.y = 10
        trj1.x_v = 0.05
        trj1.y_v = 0.02
        self.trjs.append(trj1)

        trj2 = trajectory()
        trj2.x = 100
        trj2.y = 0
        trj2.x_v = -0.005
        trj2.y_v = 0.01
        self.trjs.append(trj2)

        self.num_objects = len(self.trjs)

        # Predict the positions for each trajectory.
        self.trj_pos = self._make_pos_from_trajectories(self.trjs)

        # Put the known objects into a dictionary.
        self.known_objects = {}
        for i in range(self.num_objects):
            self.known_objects[i] = self.trj_pos[i]

    def test_overlaps(self):
        # Full overlap
        c = count_known_objects_found(self.known_objects, self.trj_pos,
                                      0.001, self.num_time_steps)
        self.assertEqual(c, self.num_objects)

        # Partial overlap
        found = [self.trj_pos[0], self.trj_pos[2]]
        c = count_known_objects_found(self.known_objects, found,
                                      0.001, self.num_time_steps)
        self.assertEqual(c, 2)

    def test_nones(self):
        found = self._make_pos_from_trajectories(self.trjs)

        # Add None observations: 1 to 0, 2 to 1, and 5 to 2.
        found[0][1] = None
        found[1][2] = None
        found[1][4] = None
        found[2][0] = None
        found[2][1] = None
        found[2][2] = None
        found[2][8] = None
        found[2][9] = None

        # No matches at num_matches = 10
        c = count_known_objects_found(self.known_objects, found,
                                      0.1, self.num_time_steps)
        self.assertEqual(c, 0)

        # One match at num_matches = 9
        c = count_known_objects_found(self.known_objects, found,
                                      0.1, 9)
        self.assertEqual(c, 1)

        # Two matches at num_matches = 8
        c = count_known_objects_found(self.known_objects, found,
                                      0.1, 8)
        self.assertEqual(c, 2)

    def test_thresholds(self):
        # Move the starting location of trajectory 0.
        self.trjs[0].x = 55
        self.trjs[0].y = 55

        # We only find 2 matches.
        found = self._make_pos_from_trajectories(self.trjs)
        c = count_known_objects_found(self.known_objects, found,
                                      0.1, self.num_time_steps)
        self.assertEqual(c, 2)

        # Move the velocity of trajectory 2.
        self.trjs[2].x_v = 0.05

        # We only find 1 match.
        found = self._make_pos_from_trajectories(self.trjs)
        c = count_known_objects_found(self.known_objects, found,
                                      0.1, self.num_time_steps)
        self.assertEqual(c, 1)

        # Move the velocity of trajectory 1 less
        # than the threshold.
        self.trjs[1].y_v = 0.020000001

        # We still find 1 match.
        found = self._make_pos_from_trajectories(self.trjs)
        c = count_known_objects_found(self.known_objects, found,
                                      0.001, self.num_time_steps)
        self.assertEqual(c, 1)

if __name__ == '__main__':
   unittest.main()

