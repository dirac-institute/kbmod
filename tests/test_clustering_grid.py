import unittest

from kbmod.filters.clustering_grid import TrajectoryClusterGrid
from kbmod.search import Trajectory


class test_trajectory_cluster_grid(unittest.TestCase):
    def test_trajectory_cluster_grid_basic(self):
        """Tests the basic functionality of the TrajectoryClusterGrid."""
        table = TrajectoryClusterGrid(10, 1.0)
        self.assertEqual(len(table), 0)
        self.assertEqual(table.total_count, 0)

        # Add a single trajectory.
        table.add_trajectory(Trajectory(0, 0, 0.0, 0.0, 1.0, 10.0, 10))
        self.assertEqual(len(table), 1)
        self.assertEqual(table.total_count, 1)
        self.assertIsNotNone(table.table.get((0, 0, 0, 0)))
        self.assertEqual(table.count.get((0, 0, 0, 0)), 1)
        self.assertEqual(table.get_indices(), [0])

        # Add a few more trajectories.  The first two are new bins and the
        # third overlaps the second.
        table.add_trajectory(Trajectory(21, 21, 10.0, 10.0, 1.0, 10.0, 10))
        table.add_trajectory(Trajectory(21, 21, 0.0, 0.0, 1.0, 10.0, 10))
        table.add_trajectory(Trajectory(21, 21, 0.0, 0.0, 1.0, 100.0, 9))

        self.assertEqual(len(table), 3)
        self.assertEqual(table.total_count, 4)
        self.assertIsNotNone(table.table.get((0, 0, 0, 0)))
        self.assertIsNotNone(table.table.get((2, 2, 3, 3)))
        self.assertIsNotNone(table.table.get((2, 2, 2, 2)))
        self.assertEqual(table.count.get((0, 0, 0, 0)), 1)
        self.assertEqual(table.count.get((2, 2, 3, 3)), 1)
        self.assertEqual(table.count.get((2, 2, 2, 2)), 2)
        self.assertEqual(table.table[(2, 2, 2, 2)].obs_count, 9)

        self.assertEqual(set(table.get_indices()), set([0, 1, 3]))
        self.assertEqual(len(table.get_trajectories()), 3)

        # Add a worse trajectory to the (0, 0, 0, 0) bin.
        table.add_trajectory(Trajectory(0, 0, 0.0, 0.0, 1.0, 5.0, 5))
        self.assertEqual(len(table), 3)
        self.assertEqual(table.count[(0, 0, 0, 0)], 2)
        self.assertEqual(table.table[(0, 0, 0, 0)].obs_count, 10)

        self.assertEqual(set(table.get_indices()), set([0, 1, 3]))
        self.assertEqual(len(table.get_trajectories()), 3)

        # Add a better trajectory to the (0, 0, 0, 0) bin.
        table.add_trajectory(Trajectory(0, 0, 0.0, 0.0, 1.0, 15.0, 15), idx=10)
        self.assertEqual(len(table), 3)
        self.assertEqual(table.count[(0, 0, 0, 0)], 3)
        self.assertEqual(table.table[(0, 0, 0, 0)].obs_count, 15)

        self.assertEqual(set(table.get_indices()), set([10, 1, 3]))
        self.assertEqual(len(table.get_trajectories()), 3)


if __name__ == "__main__":
    unittest.main()
