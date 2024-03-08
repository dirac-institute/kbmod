import unittest

from kbmod.search import HAS_GPU, Trajectory, TrajectoryList
from kbmod.trajectory_utils import make_trajectory


class test_trajectory_list(unittest.TestCase):
    def setUp(self):
        self.max_size = 10
        self.trj_list = TrajectoryList(self.max_size)
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, make_trajectory(x=i))

    def test_create(self):
        self.assertFalse(self.trj_list.on_gpu)
        self.assertEqual(self.trj_list.get_size(), self.max_size)
        self.assertEqual(len(self.trj_list), self.max_size)
        for i in range(self.max_size):
            self.assertIsNotNone(self.trj_list.get_trajectory(i))
        self.assertEqual(len(self.trj_list.get_list()), self.max_size)

        # Cannot create a zero or negative length list.
        self.assertRaises(RuntimeError, TrajectoryList, -1)

    def test_resize(self):
        # Resizing down drops values at the end.
        self.trj_list.resize(5)
        self.assertEqual(self.trj_list.get_size(), 5)
        for i in range(5):
            self.assertEqual(self.trj_list.get_trajectory(i).x, i)

        # Resizing up adds values at the end.
        self.trj_list.resize(8)
        self.assertEqual(self.trj_list.get_size(), 8)
        for i in range(5):
            self.assertEqual(self.trj_list.get_trajectory(i).x, i)
        for i in range(5, 8):
            self.assertEqual(self.trj_list.get_trajectory(i).x, 0)

    def test_get_set(self):
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, make_trajectory(y=i))
        for i in range(self.max_size):
            self.assertEqual(self.trj_list.get_trajectory(i).y, i)

        # The retrieved trajectories are modifiable
        trj = self.trj_list.get_trajectory(1)
        trj.x = 101
        self.assertEqual(self.trj_list.get_trajectory(1).x, 101)

        # Cannot get or set out of bounds.
        self.assertRaises(RuntimeError, self.trj_list.get_trajectory, self.max_size + 1)
        self.assertRaises(RuntimeError, self.trj_list.get_trajectory, -1)

        new_trj = make_trajectory(x=10)
        self.assertRaises(RuntimeError, self.trj_list.set_trajectory, self.max_size + 1, new_trj)
        self.assertRaises(RuntimeError, self.trj_list.set_trajectory, -1, new_trj)

    def test_get_batch(self):
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, make_trajectory(x=i))
        subset = self.trj_list.get_batch(3, 2)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0].x, 3)
        self.assertEqual(subset[1].x, 4)

        # We can run off the end.
        subset = self.trj_list.get_batch(3, 100)
        self.assertEqual(len(subset), 7)

        # We cannot use an invalid starting index or batch size
        self.assertRaises(RuntimeError, self.trj_list.get_batch, -1, 3)
        self.assertRaises(RuntimeError, self.trj_list.get_batch, 3, -1)

    def test_sort(self):
        lh = [100.0, 110.0, 90.0, 120.0, 125.0]
        obs_count = [10, 9, 8, 6, 7]
        lh_order = [4, 3, 1, 0, 2]
        obs_order = [0, 1, 2, 4, 3]

        trjs = TrajectoryList(5)
        for i in range(5):
            trj = make_trajectory(x=i, lh=lh[i], obs_count=obs_count[i])
            trjs.set_trajectory(i, trj)

        trjs.sort_by_likelihood()
        for i in range(5):
            self.assertEqual(trjs.get_trajectory(i).x, lh_order[i])

        trjs.sort_by_obs_count()
        for i in range(5):
            self.assertEqual(trjs.get_trajectory(i).x, obs_order[i])

    def test_filter_on_lh(self):
        lh = [100.0, 110.0, 90.0, 120.0, 125.0, 121.0, 10.0]
        trjs = TrajectoryList(len(lh))
        for i in range(len(lh)):
            trjs.set_trajectory(i, make_trajectory(x=i, lh=lh[i]))

        trjs.filter_by_likelihood(110.0)
        expected = set([4, 5, 3, 1])

        # Test that each remaining result appears once in the expected set.
        self.assertEqual(len(trjs), len(expected))
        for i in range(len(trjs)):
            idx = trjs.get_trajectory(i).x
            self.assertTrue(idx in expected)
            expected.remove(idx)

    def test_filter_on_obs_count(self):
        vals = [10, 7, 8, 9, 12, 15, 1, 2, 19, 3]
        trjs = TrajectoryList(len(vals))
        for i in range(len(vals)):
            trjs.set_trajectory(i, make_trajectory(x=i, obs_count=vals[i]))

        trjs.filter_by_obs_count(10)
        expected = set([8, 5, 4, 0])

        # Test that each remaining result appears once in the expected set.
        self.assertEqual(len(trjs), len(expected))
        for i in range(len(trjs)):
            idx = trjs.get_trajectory(i).x
            self.assertTrue(idx in expected)
            expected.remove(idx)

    def test_filter_on_valid(self):
        vals = [True, False, False, True, True, False, True, True, False]
        trjs = TrajectoryList(len(vals))
        for i in range(len(vals)):
            trj = make_trajectory(x=i)
            trj.valid = vals[i]
            trjs.set_trajectory(i, trj)

        trjs.filter_by_valid()
        expected = set([0, 3, 4, 6, 7])

        # Test that each remaining result appears once in the expected set.
        self.assertEqual(len(trjs), len(expected))
        for i in range(len(trjs)):
            idx = trjs.get_trajectory(i).x
            self.assertTrue(idx in expected)
            expected.remove(idx)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_move_to_from_gpu(self):
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, make_trajectory(x=i))

        # Move to GPU.
        self.trj_list.move_to_gpu()
        self.assertTrue(self.trj_list.on_gpu)

        # We cannot get or set.
        self.assertRaises(RuntimeError, self.trj_list.get_trajectory, 0)

        new_trj = make_trajectory(x=10)
        self.assertRaises(RuntimeError, self.trj_list.set_trajectory, 0, new_trj)

        # Moving to GPU again does not do anything.
        self.trj_list.move_to_gpu()
        self.assertTrue(self.trj_list.on_gpu)

        # Move back to CPU.
        self.trj_list.move_to_cpu()
        self.assertFalse(self.trj_list.on_gpu)
        self.trj_list.set_trajectory(0, new_trj)
        self.assertEqual(self.trj_list.get_trajectory(0).x, 10)

        # Moving back to CPU again doesn't do anything.
        self.trj_list.move_to_cpu()
        self.assertFalse(self.trj_list.on_gpu)


if __name__ == "__main__":
    unittest.main()
