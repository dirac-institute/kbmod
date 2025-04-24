import unittest

from kbmod.search import HAS_GPU, Trajectory, TrajectoryList


class test_trajectory_list(unittest.TestCase):
    def setUp(self):
        self.max_size = 10
        self.trj_list = TrajectoryList(self.max_size)
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, Trajectory(x=i))

    def test_create(self):
        self.assertFalse(self.trj_list.on_gpu)
        self.assertEqual(self.trj_list.get_size(), self.max_size)
        self.assertEqual(len(self.trj_list), self.max_size)
        for i in range(self.max_size):
            self.assertIsNotNone(self.trj_list.get_trajectory(i))
        self.assertEqual(len(self.trj_list.get_list()), self.max_size)

        # Create from a list
        trj_list2 = TrajectoryList([Trajectory(x=2 * i) for i in range(8)])
        self.assertEqual(trj_list2.get_size(), 8)
        for i in range(8):
            self.assertEqual(trj_list2.get_trajectory(i).x, 2 * i)

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
            self.trj_list.set_trajectory(i, Trajectory(y=i))
        for i in range(self.max_size):
            self.assertEqual(self.trj_list.get_trajectory(i).y, i)

        # The retrieved trajectories are modifiable
        trj = self.trj_list.get_trajectory(1)
        trj.x = 101
        self.assertEqual(self.trj_list.get_trajectory(1).x, 101)

        # Cannot get or set out of bounds.
        self.assertRaises(RuntimeError, self.trj_list.get_trajectory, self.max_size + 1)

        new_trj = Trajectory(x=10)
        self.assertRaises(RuntimeError, self.trj_list.set_trajectory, self.max_size + 1, new_trj)

    def test_get_batch(self):
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, Trajectory(x=i))
        subset = self.trj_list.get_batch(3, 2)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[0].x, 3)
        self.assertEqual(subset[1].x, 4)

        # Get exactly the max number.
        subset = self.trj_list.get_batch(0, self.max_size)
        self.assertEqual(len(subset), self.max_size)
        for i in range(self.max_size):
            self.assertEqual(subset[i].x, i)

        # We can run off the end.
        subset = self.trj_list.get_batch(5, 100)
        self.assertEqual(len(subset), self.max_size - 5)
        for i in range(5, self.max_size):
            self.assertEqual(subset[i - 5].x, i)

    def test_sort(self):
        lh = [100.0, 110.0, 90.0, 120.0, 125.0]
        obs_count = [10, 9, 8, 6, 7]
        lh_order = [4, 3, 1, 0, 2]
        obs_order = [0, 1, 2, 4, 3]

        trjs = TrajectoryList(5)
        for i in range(5):
            trj = Trajectory(x=i, lh=lh[i], obs_count=obs_count[i])
            trjs.set_trajectory(i, trj)

        trjs.sort_by_likelihood()
        for i in range(5):
            self.assertEqual(trjs.get_trajectory(i).x, lh_order[i])

    def test_filter_by_lh(self):
        lh = [100.0, 110.0, 90.0, 120.0, 125.0]
        obs_count = [10, 9, 8, 6, 7]

        trjs = TrajectoryList(5)
        for i in range(5):
            trj = Trajectory(x=i, lh=lh[i], obs_count=obs_count[i])
            trjs.set_trajectory(i, trj)
        self.assertEqual(len(trjs), 5)

        trjs.filter_by_likelihood(110.0)
        self.assertEqual(len(trjs), 3)
        self.assertEqual(set([trjs.get_trajectory(i).x for i in range(3)]), set([1, 3, 4]))

    def test_filter_by_obs_count(self):
        lh = [100.0, 110.0, 90.0, 120.0, 125.0, 120.0]
        obs_count = [10, 9, 8, 6, 7, 11]

        trjs = TrajectoryList(6)
        for i in range(6):
            trj = Trajectory(x=i, lh=lh[i], obs_count=obs_count[i])
            trjs.set_trajectory(i, trj)
        self.assertEqual(len(trjs), 6)

        trjs.filter_by_obs_count(8)
        self.assertEqual(len(trjs), 4)
        self.assertEqual(set([trjs.get_trajectory(i).x for i in range(4)]), set([0, 1, 2, 5]))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_move_to_from_gpu(self):
        for i in range(self.max_size):
            self.trj_list.set_trajectory(i, Trajectory(x=i))

        # Move to GPU.
        self.trj_list.move_to_gpu()
        self.assertTrue(self.trj_list.on_gpu)

        # We cannot get or set.
        self.assertRaises(RuntimeError, self.trj_list.get_trajectory, 0)

        new_trj = Trajectory(x=10)
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
