import unittest

from kbmod.trajectory_generator import (
    KBMODV1Search,
    SingleVelocitySearch,
    RandomVelocitySearch,
    VelocityGridSearch,
)


class test_trajectory_generator(unittest.TestCase):
    def test_SingleVelocitySearch(self):
        gen = SingleVelocitySearch(10.0, 5.0)
        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 1)
        self.assertEqual(trjs[0].vx, 10.0)
        self.assertEqual(trjs[0].vy, 5.0)

    def test_VelocityGridSearch(self):
        gen = VelocityGridSearch(3, 0.0, 2.0, 3, -0.25, 0.25)
        expected_x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
        expected_y = [-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25]

        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 9)
        for i in range(6):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test invalid number of steps or ranges.
        self.assertRaises(ValueError, VelocityGridSearch, 3, 0.0, 2.0, 0, -0.25, 0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 0, 0.0, 2.0, 3, -0.25, 0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 3, 0.0, 2.0, 3, 0.25, -0.25)
        self.assertRaises(ValueError, VelocityGridSearch, 3, 2.0, 0.0, 3, -0.25, 0.25)

    def test_KBMODV1Search(self):
        # Note that KBMOD v1's search will never include the upper bound of angle or velocity.
        gen = KBMODV1Search(3, 0.0, 3.0, 2, -0.25, 0.25)
        expected_x = [0.0, 0.9689, 1.9378, 0.0, 1.0, 2.0]
        expected_y = [0.0, -0.247, -0.4948, 0.0, 0.0, 0.0]

        trjs = [trj for trj in gen]
        self.assertEqual(len(trjs), 6)
        for i in range(6):
            self.assertAlmostEqual(trjs[i].vx, expected_x[i], delta=0.001)
            self.assertAlmostEqual(trjs[i].vy, expected_y[i], delta=0.001)

        # Test invalid number of steps.
        self.assertRaises(ValueError, KBMODV1Search, 3, 0.0, 3.0, 0, -0.25, 0.25)
        self.assertRaises(ValueError, KBMODV1Search, 0, 0.0, 3.0, 2, -0.25, 0.25)
        self.assertRaises(ValueError, KBMODV1Search, 3, 0.0, 3.0, 2, 0.25, -0.25)
        self.assertRaises(ValueError, KBMODV1Search, 3, 3.5, 3.0, 2, -0.25, 0.25)

    def test_RandomVelocitySearch(self):
        gen = RandomVelocitySearch(0.0, 2.0, -0.25, 0.25)

        # Try at least 1000 iterations and make sure it is still generating.
        for itr in range(1000):
            trj = next(gen)
            self.assertGreaterEqual(trj.x, 0.0)
            self.assertLessEqual(trj.x, 2.0)
            self.assertGreaterEqual(trj.y, -0.25)
            self.assertLessEqual(trj.y, 0.25)

        # Generate a single additional candidate.
        gen2 = RandomVelocitySearch(0.0, 2.0, -0.25, 0.25, max_samples=1)
        self.assertEqual(len([trj for trj in gen2]), 1)

        # No more samples to generate
        self.assertEqual(len([trj for trj in gen2]), 0)

        # Generate a twenty more candidates.
        gen2.reset_sample_count(20)
        self.assertEqual(len([trj for trj in gen2]), 20)


if __name__ == "__main__":
    unittest.main()
