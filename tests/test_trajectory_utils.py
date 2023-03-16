import math
import unittest

from kbmod.search import *


class test_predicted_position(unittest.TestCase):
    def setUp(self):
        # create a trajectory for the object
        self.trj = trajectory()
        self.trj.x = 10
        self.trj.y = 11
        self.trj.x_v = 1.0
        self.trj.y_v = -0.5

        self.trj2 = trajectory()
        self.trj2.x = 11
        self.trj2.y = 10
        self.trj2.x_v = 0
        self.trj2.y_v = 0

    def test_trajectory_result_default(self):
        res = trj_result(self.trj, 10)
        inds = res.get_valid_indices_list()
        self.assertEqual(res.num_times(), 10)
        self.assertEqual(len(inds), 10)
        for i in range(10):
            self.assertTrue(res.check_index_valid(i))
            self.assertEqual(inds[i], i)

        # Set 3 and 6 False
        res.set_index_valid(3, False)
        res.set_index_valid(6, False)

        # Check that the False indices are showing as False.
        inds = res.get_valid_indices_list()
        self.assertEqual(res.num_times(), 10)
        self.assertEqual(len(inds), 8)
        for i in range(10):
            if i != 3 and i != 6:
                self.assertTrue(res.check_index_valid(i))
                self.assertTrue(i in inds)
            else:
                self.assertFalse(res.check_index_valid(i))
                self.assertFalse(i in inds)

    def test_trajectory_result_indices(self):
        valid = [1, 2, 5, 6, 7]
        res = trj_result(self.trj, 10, valid)
        inds = res.get_valid_indices_list()
        self.assertEqual(res.num_times(), 10)
        self.assertEqual(len(inds), len(valid))
        for i in range(10):
            self.assertEqual(res.check_index_valid(i), i in valid)
            self.assertEqual(i in inds, i in valid)

    def test_trajectory_result_boolean(self):
        valid = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        res = trj_result(self.trj, valid)
        inds = res.get_valid_indices_list()
        self.assertEqual(res.num_times(), 10)
        self.assertEqual(len(inds), 5)
        for i in range(10):
            self.assertEqual(res.check_index_valid(i), valid[i] == 1)
            self.assertEqual(i in inds, valid[i] == 1)

    def test_prediction(self):
        p = compute_traj_pos(self.trj, 0)
        self.assertAlmostEqual(p.x, self.trj.x, delta=1e-5)
        self.assertAlmostEqual(p.y, self.trj.y, delta=1e-5)

        p = compute_traj_pos(self.trj, 0.5)
        self.assertAlmostEqual(p.x, self.trj.x + 0.5 * self.trj.x_v, delta=1e-5)
        self.assertAlmostEqual(p.y, self.trj.y + 0.5 * self.trj.y_v, delta=1e-5)

    def test_prediction_bc(self):
        bc = baryCorrection()
        bc.dx = 0.1
        bc.dxdx = 0.01
        bc.dxdy = 0.02
        bc.dy = 0.05
        bc.dydx = 0.005
        bc.dydy = 0.01

        p = compute_traj_pos_bc(self.trj, 0, bc)
        true_x = self.trj.x + 0.1 + 0.01 * self.trj.x + 0.02 * self.trj.y
        self.assertAlmostEqual(p.x, true_x, delta=1e-5)
        true_y = self.trj.y + 0.05 + 0.005 * self.trj.x + 0.01 * self.trj.y
        self.assertAlmostEqual(p.y, true_y, delta=1e-5)

        p = compute_traj_pos_bc(self.trj, 2.0, bc)
        true_x = self.trj.x + 2.0 * self.trj.x_v + 0.1 + 0.01 * self.trj.x + 0.02 * self.trj.y
        self.assertAlmostEqual(p.x, true_x, delta=1e-5)
        true_y = self.trj.y + 2.0 * self.trj.y_v + 0.05 + 0.005 * self.trj.x + 0.01 * self.trj.y
        self.assertAlmostEqual(p.y, true_y, delta=1e-5)

    def test_prediction_bc_0(self):
        bc = baryCorrection()
        bc.dx = 0.0
        bc.dxdx = 0.0
        bc.dxdy = 0.0
        bc.dy = 0.0
        bc.dydx = 0.0
        bc.dydy = 0.0

        p1 = compute_traj_pos(self.trj, 15.0)
        p2 = compute_traj_pos_bc(self.trj, 15.0, bc)

        # With all BaryCorr coefficients set to zero the predictions
        # should be indentical.
        self.assertAlmostEqual(p1.x, p2.x, delta=1e-5)
        self.assertAlmostEqual(p1.y, p2.y, delta=1e-5)

    def test_ave_distance(self):
        posA = [compute_traj_pos(self.trj, 0.0)]
        posB = [compute_traj_pos(self.trj2, 0.0)]
        result = ave_trajectory_dist(posA, posB)
        self.assertAlmostEqual(result, math.sqrt(2.0), delta=1e-5)

        result = ave_trajectory_dist(posB, posA)
        self.assertAlmostEqual(result, math.sqrt(2.0), delta=1e-5)

        posA.append(compute_traj_pos(self.trj, 1.0))
        posB.append(compute_traj_pos(self.trj2, 1.0))
        result = ave_trajectory_dist(posA, posB)
        dist = (math.sqrt(2.0) + 0.5) / 2.0
        self.assertAlmostEqual(result, dist, delta=1e-5)

        posA.append(compute_traj_pos(self.trj, 2.0))
        posB.append(compute_traj_pos(self.trj2, 2.0))
        result = ave_trajectory_dist(posA, posB)
        dist = (math.sqrt(2.0) + 0.5 + 1.0) / 3.0
        self.assertAlmostEqual(result, dist, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
