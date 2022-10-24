import tempfile
import unittest

import numpy as np
from analysis_utils import PostProcess

from kbmod import *


class test_kernels_wrappers(unittest.TestCase):
    def setUp(self):
        # Set up initial variables for analysis_utils.PostProcess
        self.config = {}
        self.config["num_cores"] = 1
        self.config["sigmaG_lims"] = [0.0, 1.0]
        self.config["eps"] = None
        self.config["cluster_type"] = None
        self.config["cluster_function"] = None
        self.config["clip_negative"] = None
        self.config["mask_bits_dict"] = None
        self.config["flag_keys"] = None
        self.config["repeated_flag_keys"] = None

        # Set up old_results object for analysis_utils.PostProcess
        self.num_curves = 4
        # First 3 passing indices
        psi_curves = [np.array([1.0 + (x / 100) for x in range(20)]) for _ in range(self.num_curves - 1)]
        phi_curves = [np.array([1.0 + (y / 100) for y in range(20)]) for _ in range(self.num_curves - 1)]
        # Failing index
        psi_curves.append(np.array([-200.0 - (100.0 * z) for z in range(20)]))
        phi_curves.append(np.array([1.0 for _ in range(20)]))
        # Original likelihood
        results = [1.0 for _ in range(self.num_curves)]

        self.old_results = {}
        self.old_results["psi_curves"] = psi_curves
        self.old_results["phi_curves"] = phi_curves
        self.old_results["results"] = results

        # test pass thresholds
        self.pixel_error = 0
        self.velocity_error = 0.05
        self.flux_error = 0.15

        # image properties
        self.imCount = 3
        self.dim_x = 80
        self.dim_y = 60
        self.noise_level = 8.0
        self.variance = self.noise_level**2
        self.p = psf(1.0)

        # object properties
        self.object_flux = 250.0
        self.start_x = 17
        self.start_y = 12
        self.x_vel = 21.0
        self.y_vel = 16.0

        # create a trajectory for the object
        self.trj = trajectory()
        self.trj.x = self.start_x
        self.trj.y = self.start_y
        self.trj.x_v = self.x_vel
        self.trj.y_v = self.y_vel

        # search parameters
        self.angle_steps = 10
        self.velocity_steps = 10
        self.min_angle = 0.0
        self.max_angle = 1.5
        self.min_vel = 5.0
        self.max_vel = 40.0

        # Select one pixel to mask in every other image.
        self.masked_x = 5
        self.masked_y = 6

        self.imlist = []
        for i in range(self.imCount):
            time = i / self.imCount
            im = layered_image(str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p)
            im.add_object(
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
            )

            # Mask a pixel in half the images.
            if i % 2 == 0:
                mask = im.get_mask()
                mask.set_pixel(self.masked_x, self.masked_y, 1)
                im.set_mask(mask)
                im.apply_mask_flags(1, [])

            self.imlist.append(im)

        self.config["lh_level"] = 10.0
        self.config["chunk_size"] = 5
        self.config["filter_type"] = "kalman"
        self.config["max_lh"] = 1000.0

    def test_sigmag_filtered_indices_same(self):
        # With everything the same, nothing should be filtered.
        values = [1.0 for _ in range(20)]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), 20)

    def test_sigmag_filtered_indices_no_outliers(self):
        # Try with a median of 1.0 and a percentile range of 3.0 (2.0 - -1.0).
        # It should filter any values outside [-3.45, 5.45]
        values = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.1]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), len(values))

    def test_sigmag_filtered_indices_one_outlier(self):
        # Try with a median of 1.0 and a percentile range of 3.0 (2.0 - -1.0).
        # It should filter any values outside [-3.45, 5.45]
        values = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 5.46]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), len(values) - 1)

        # The only missing value should be index=8.
        for i in range(8):
            self.assertTrue(i in inds)
        self.assertFalse(8 in inds)

        # All points pass if we use a larger width.
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 3.0)
        self.assertEqual(len(inds), len(values))

    def test_sigmag_filtered_indices_other_bounds(self):
        # Do the filtering of test_sigmag_filtered_indices_one_outlier
        # with wider bounds [-1.8944, 3.8944].
        values = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.85]
        inds = sigmag_filtered_indices(values, 0.15, 0.85, 0.4824, 2.0)

        # Nothing is filtered this time.
        self.assertEqual(len(inds), len(values))
        for i in range(9):
            self.assertTrue(i in inds)

        # Move one of the points to be an outlier.
        values = [-1.9, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.85]
        inds = sigmag_filtered_indices(values, 0.15, 0.85, 0.4824, 2.0)

        # The first entry is filtered this time.
        self.assertEqual(len(inds), len(values) - 1)
        self.assertFalse(0 in inds)
        for i in range(1, 9):
            self.assertTrue(i in inds)

    def test_sigmag_filtered_indices_two_outliers(self):
        # Try with a median of 0.0 and a percentile range of 1.1 (1.0 - -0.1).
        # It should filter any values outside [-1.631, 1.631].
        values = [1.6, 0.0, 1.0, 0.0, -1.5, 0.5, 1000.1, 0.0, 0.0, -5.2, -0.1]
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        for idx in inds:
            self.assertGreaterEqual(values[idx], -1.631)
            self.assertLessEqual(values[idx], 1.631)
        self.assertEqual(len(inds), len(values) - 2)

        # One more point passes if we use a larger width.
        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 20.0)
        self.assertEqual(len(inds), len(values) - 1)

    def test_sigmag_filtered_indices_three_outliers(self):
        # Try with a median of 5.0 and a percentile range of 4.0 (7.0-3.0).
        # It should filter any values outside [-0.93, 10.93].
        values = [5.0]
        for i in range(12):
            values.append(3.0)
        values.append(10.95)
        values.append(-1.50)
        for i in range(12):
            values.append(7.0)
        values.append(-0.95)
        values.append(7.0)

        inds = sigmag_filtered_indices(values, 0.25, 0.75, 0.7413, 2.0)
        self.assertEqual(len(inds), len(values) - 3)

        for i in range(29):
            valid = i != 13 and i != 14 and i != 27
            self.assertEqual(i in inds, valid)

    def test_kalman_filtered_indices(self):
        # With everything the same, nothing should be filtered.
        psi_values = [[1.0 for _ in range(20)] for _ in range(20)]
        phi_values = [[1.0 for _ in range(20)] for _ in range(20)]

        inds = kalman_filtered_indices(psi_values, phi_values)

        self.assertEqual(len(inds), 20)
        for i in inds:
            self.assertFalse(i[1] == [-1])
            self.assertEqual(i[2], 4.47213595499958)

    def test_kalman_filtered_indices_zero_phi(self):
        # make sure kalman filtering properly handles when a phi
        # value is 0. The last value should be masked and filtered out.
        psi_values = [[1.0 for _ in range(20)] for _ in range(21)]
        phi_values = [[1.0 for _ in range(20)] for _ in range(20)]
        phi_values.append([0.0 for _ in range(20)])

        inds = kalman_filtered_indices(psi_values, phi_values)

        self.assertEqual(len(inds), 21)
        for i in inds[:-1]:
            self.assertFalse(i[1] == [-1])
            self.assertEqual(i[2], 4.47213595499958)
        self.assertEqual(inds[-1][2], 0.0)

    def test_kalman_filtered_indices_negative_phi(self):
        # make sure kalman filtering properly handles when a phi
        # value is less than -999.0. The last value should be
        # xmasked and filtered out.
        psi_values = [[1.0 for _ in range(20)] for _ in range(21)]
        phi_values = [[1.0 for _ in range(20)] for _ in range(20)]
        phi_values.append([-999.1 for _ in range(20)])

        inds = kalman_filtered_indices(psi_values, phi_values)

        self.assertEqual(len(inds), 21)
        for i in inds[:-1]:
            self.assertFalse(i[1] == [-1])
            self.assertEqual(i[2], 4.47213595499958)
        self.assertEqual(inds[-1][2], 0.0)

    def test_kalman_filtered_indices_negative_flux(self):
        # make sure kalman filtering filters out all indices with
        # a negative flux value.
        psi_values = [[-1.0 for _ in range(20)] for _ in range(20)]
        phi_values = [[1.0 for _ in range(20)] for _ in range(20)]
        phi_values.append([-999.1 for _ in range(20)])

        inds = kalman_filtered_indices(psi_values, phi_values)

        self.assertEqual(len(inds), 20)
        for i in inds[:-1]:
            self.assertTrue(i[1] == [-1])
            self.assertEqual(i[2], 0.0)

    def test_kalman_filtered_indices_bright_first_obs(self):
        # make sure kalman filtering doesn't discard a potential
        # trajectory just because the first flux value is extra
        # bright (testing the reversed kalman flux calculation).
        psi_values = [[1000.0, 1.0, 1.0, 1.0, 1]]
        phi_values = [[1.0, 1.0, 1.0, 1.0, 1]]

        inds = kalman_filtered_indices(psi_values, phi_values)

        self.assertEqual(inds[0][2], 2.0)

    def test_kalman_filtered_indices_in_the_middle(self):
        # make sure kalman filtering can reject indexs in
        # the middle of the values arrays
        psi_values = [[1.0 for _ in range(5)] for _ in range(20)]
        phi_values = [[1.0 for _ in range(5)] for _ in range(20)]

        psi_values[10] = [1.0, 1.0, 1.0, 1.0, 1.0]
        phi_values[10] = [1.0, 1.0, 100000000.0, 1.0, 1]

        inds = kalman_filtered_indices(psi_values, phi_values)

        self.assertEqual(len(inds), 20)
        for i in inds[:10]:
            self.assertFalse(i[1] == [-1])
            self.assertEqual(i[2], 2.23606797749979)

        self.assertTrue(inds[10][2] < 0.0005)

        for i in inds[11:]:
            self.assertFalse(i[1] == [-1])
            self.assertEqual(i[2], 2.23606797749979)

    def test_calculate_likelihood_psiphi(self):
        # make sure that the calculate_likelihood_psi_phi works.
        psi_values = [1.0 for _ in range(20)]
        phi_values = [1.0 for _ in range(20)]

        lh = calculate_likelihood_psi_phi(psi_values, phi_values)

        self.assertEqual(lh, 4.47213595499958)

    def test_calculate_likelihood_psiphi_zero_or_negative_phi(self):
        # make sure that the calculate_likelihood_psi_phi works
        # properly when phi values are less than or equal to zero.
        psi_values = [1.0 for _ in range(20)]
        phi_values = [-1.0 for _ in range(20)]

        # test negatives
        lh = calculate_likelihood_psi_phi(psi_values, phi_values)
        self.assertEqual(lh, 0.0)

        # test zero
        lh = calculate_likelihood_psi_phi([1.0], [0.0])
        self.assertEqual(lh, 0.0)

    def test_apply_clipped_average_single_thread(self):
        # make sure apply_clipped_average works when num_cores == 1
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_average(self.old_results, None, {}, 0.5)

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(res[self.num_curves - 1][1], [-1])

    def test_apply_clipped_average_multi_thread(self):
        # make sure apply_clipped_average works when multithreading is enabled
        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_average(self.old_results, None, {}, 0.5)

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(res[self.num_curves - 1][1], [-1])

    def test_apply_clipped_sigmaG_single_thread(self):
        # make sure apply_clipped_sigmaG works when num_cores == 1
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_sigmaG(self.old_results, None, {"sigmaG_filter_type": "lh"}, 0.5)

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(res[self.num_curves - 1][1], [-1])

    def test_apply_clipped_sigmaG_multi_thread(self):
        # make sure apply_clipped_sigmaG works when multithreading is enabled
        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_sigmaG(self.old_results, None, {"sigmaG_filter_type": "lh"}, 0.5)

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(res[self.num_curves - 1][1], [-1])

    def test_apply_stamp_filter_single_thread(self):
        # make sure apply_stamp_filter works when num_cores == 1
        stack = image_stack(self.imlist)
        search = stack_search(stack)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

        kb_post_process = PostProcess(self.config)

        mjds = np.array(stack.get_times())
        keep = kb_post_process.load_results(
            search,
            mjds,
            {},
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            filter_type=self.config["filter_type"],
            max_lh=self.config["max_lh"],
        )

        res = kb_post_process.apply_stamp_filter(keep, search)

        self.assertIsNotNone(res["stamps"])
        self.assertIsNotNone(res["final_results"])

    def test_apply_stamp_filter_multi_thread(self):
        # make sure apply_stamp_filter works when multithreading is enabled
        stack = image_stack(self.imlist)
        search = stack_search(stack)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config)

        mjds = np.array(stack.get_times())
        keep = kb_post_process.load_results(
            search,
            mjds,
            {},
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            filter_type=self.config["filter_type"],
            max_lh=self.config["max_lh"],
        )

        res = kb_post_process.apply_stamp_filter(keep, search)

        self.assertIsNotNone(res["stamps"])
        self.assertIsNotNone(res["final_results"])


if __name__ == "__main__":
    unittest.main()
