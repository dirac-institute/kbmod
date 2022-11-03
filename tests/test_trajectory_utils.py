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

    def test_convert_depth0(self):
        tr = traj_region()
        tr.ix = 15
        tr.iy = 23
        tr.fx = 20
        tr.fy = 13
        tr.depth = 0
        tr.obs_count = 19
        tr.likelihood = 100.0
        tr.flux = 101.0

        # Convert with endTime = 10.0
        t = convert_traj_region(tr, 10.0)
        self.assertEqual(t.x, 15)
        self.assertEqual(t.y, 23)
        self.assertAlmostEqual(t.x_v, 0.5)
        self.assertAlmostEqual(t.y_v, -1.0)
        self.assertEqual(t.obs_count, 19)
        self.assertAlmostEqual(t.lh, 100.0)
        self.assertAlmostEqual(t.flux, 101.0)

        # Convert with endTime = 20.0
        t = convert_traj_region(tr, 20.0)
        self.assertEqual(t.x, 15)
        self.assertEqual(t.y, 23)
        self.assertAlmostEqual(t.x_v, 0.25)
        self.assertAlmostEqual(t.y_v, -0.5)
        self.assertEqual(t.obs_count, 19)
        self.assertAlmostEqual(t.lh, 100.0)
        self.assertAlmostEqual(t.flux, 101.0)

    def test_convert_depth2(self):
        tr = traj_region()
        tr.ix = 15
        tr.iy = 23
        tr.fx = 20
        tr.fy = 13
        tr.depth = 2
        tr.obs_count = 17
        tr.likelihood = 25.0
        tr.flux = 201.0

        # Convert with endTime = 10.0
        t = convert_traj_region(tr, 10.0)
        self.assertEqual(t.x, 60)
        self.assertEqual(t.y, 92)
        self.assertAlmostEqual(t.x_v, 2.0)
        self.assertAlmostEqual(t.y_v, -4.0)
        self.assertEqual(t.obs_count, 17)
        self.assertAlmostEqual(t.lh, 25.0)
        self.assertAlmostEqual(t.flux, 201.0)

        # Convert with endTime = 20.0
        t = convert_traj_region(tr, 20.0)
        self.assertEqual(t.x, 60)
        self.assertEqual(t.y, 92)
        self.assertAlmostEqual(t.x_v, 1.0)
        self.assertAlmostEqual(t.y_v, -2.0)

    def test_subdivide(self):
        tr = traj_region()
        tr.ix = 15
        tr.iy = 23
        tr.fx = 20
        tr.fy = 13
        tr.depth = 2

        subregions = subdivide_traj_region(tr)
        self.assertEqual(len(subregions), 16)
        for i in range(16):
            self.assertEqual(subregions[i].depth, 1)
            self.assertGreaterEqual(subregions[i].ix, tr.ix * 2.0)
            self.assertGreaterEqual(subregions[i].iy, tr.iy * 2.0)
            self.assertLessEqual(subregions[i].ix, tr.ix * 2.0 + 1.0)
            self.assertLessEqual(subregions[i].iy, tr.iy * 2.0 + 1.0)
            self.assertGreaterEqual(subregions[i].fx, tr.fx * 2.0)
            self.assertGreaterEqual(subregions[i].fy, tr.fy * 2.0)
            self.assertLessEqual(subregions[i].fx, tr.fx * 2.0 + 1.0)
            self.assertLessEqual(subregions[i].fy, tr.fy * 2.0 + 1.0)

            # Check that no two subregions are the same.
            for j in range(i + 1, 16):
                self.assertFalse(
                    abs(subregions[i].ix - subregions[j].ix) < 1e-6
                    and abs(subregions[i].iy - subregions[j].iy) < 1e-6
                    and abs(subregions[i].fx - subregions[j].fx) < 1e-6
                    and abs(subregions[i].fy - subregions[j].fy) < 1e-6
                )

    def test_filter_lh(self):
        arr = []
        for i in range(8):
            tr = traj_region()
            tr.ix = i
            tr.obs_count = 10 + i
            tr.likelihood = 100.0 + 10.0 * i
            arr.append(tr)

        # Override a few to filter.
        arr[1].obs_count = 2
        arr[1].likelihood = 200.0
        arr[5].obs_count = 20
        arr[5].likelihood = 2.0
        arr[6].obs_count = 5
        arr[6].likelihood = 2.0

        # Check that the correct ones are filtered.
        arr2 = filter_traj_regions_lh(arr, 100.0, 10)
        self.assertEqual(len(arr2), 5)
        self.assertEqual(arr2[0].ix, 0.0)
        self.assertEqual(arr2[1].ix, 2.0)
        self.assertEqual(arr2[2].ix, 3.0)
        self.assertEqual(arr2[3].ix, 4.0)
        self.assertEqual(arr2[4].ix, 7.0)


if __name__ == "__main__":
    unittest.main()
