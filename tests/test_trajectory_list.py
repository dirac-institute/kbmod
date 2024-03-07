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
        self.assertRaises(RuntimeError, TrajectoryList, 0)
        self.assertRaises(RuntimeError, TrajectoryList, -1)

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
