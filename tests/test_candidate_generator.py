import unittest

from kbmod.candidate_generator import KBMODV1Search, SingleVelocitySearch, VelocityGridSearch


class test_candidate_generator(unittest.TestCase):
    def test_SingleVelocitySearch(self):
        strategy = SingleVelocitySearch(10.0, 5.0)
        trjs = strategy.get_candidate_trajectories()
        self.assertEqual(len(trjs), 1)
        self.assertEqual(trjs[0].vx, 10.0)
        self.assertEqual(trjs[0].vy, 5.0)

    def test_VelocityGridSearch(self):
        strategy = VelocityGridSearch(3, 0.0, 2.0, 3, -0.25, 0.25)
        expected_x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
        expected_y = [-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25]
        trjs = strategy.get_candidate_trajectories()
        self.assertEqual(len(trjs), 9)
        for i in range(6):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test invalid number of steps.
        self.assertRaises(ValueError, VelocityGridSearch, 3, 0.0, 2.0, 0, -0.25, 0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 0, 0.0, 2.0, 3, -0.25, 0.25)

    def test_KBMODV1Search(self):
        # Note that KBMOD v1's search will never include the upper bound of angle or velocity.
        strategy = KBMODV1Search(3, 0.0, 3.0, 2, -0.25, 0.25)
        expected_x = [0.0, 0.9689, 1.9378, 0.0, 1.0, 2.0]
        expected_y = [0.0, -0.247, -0.4948, 0.0, 0.0, 0.0]
        trjs = strategy.get_candidate_trajectories()
        self.assertEqual(len(trjs), 6)
        for i in range(6):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test invalid number of steps.
        self.assertRaises(ValueError, KBMODV1Search, 3, 0.0, 3.0, 0, -0.25, 0.25)
        self.assertRaises(ValueError, KBMODV1Search, 0, 0.0, 3.0, 2, -0.25, 0.25)


if __name__ == "__main__":
    unittest.main()
