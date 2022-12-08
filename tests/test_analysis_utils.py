import unittest

from kbmod.analysis_utils import *
from kbmod.search import *


class test_analysis_utils(unittest.TestCase):
    def _make_trajectory(self, x0, y0, xv, yv, lh):
        t = trajectory()
        t.x = x0
        t.y = y0
        t.x_v = xv
        t.y_v = yv
        t.lh = lh
        return t

    def setUp(self):
        # The configuration parameters.
        self.default_mask_bits_dict = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }
        self.default_flag_keys = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
        self.default_repeated_flag_keys = []
        self.config = {  # Mandatory values
            "im_filepath": None,
            "res_filepath": None,
            "time_file": None,
            # Suggested values
            "v_arr": [92.0, 526.0, 256],
            "ang_arr": [3.1415 / 15, 3.1415 / 15, 128],
            # Optional values
            "output_suffix": "search",
            "mjd_lims": None,
            "average_angle": None,
            "do_mask": True,
            "mask_num_images": 2,
            "mask_threshold": None,
            "mask_grow": 10,
            "lh_level": 10.0,
            "psf_val": 1.4,
            "num_obs": 10,
            "num_cores": 1,
            "visit_in_filename": [0, 6],
            "file_format": "{0:06d}.fits",
            "sigmaG_lims": [25, 75],
            "chunk_size": 500000,
            "max_lh": 1000.0,
            "filter_type": "clipped_sigmaG",
            "center_thresh": 0.00,
            "peak_offset": [2.0, 2.0],
            "mom_lims": [35.5, 35.5, 2.0, 0.3, 0.3],
            "stamp_type": "sum",
            "stamp_radius": 10,
            "eps": 0.03,
            "gpu_filter": False,
            "do_clustering": True,
            "do_stamp_filter": True,
            "clip_negative": False,
            "sigmaG_filter_type": "lh",
            "cluster_type": "all",
            "cluster_function": "DBSCAN",
            "mask_bits_dict": self.default_mask_bits_dict,
            "flag_keys": self.default_flag_keys,
            "repeated_flag_keys": self.default_repeated_flag_keys,
            "bary_dist": None,
            "encode_psi_bytes": -1,
            "encode_phi_bytes": -1,
            "known_obj_thresh": None,
            "known_obj_jpl": False,
        }

        # search parameters
        self.angle_steps = 10
        self.velocity_steps = 10
        self.min_angle = 0.0
        self.max_angle = 1.5
        self.min_vel = 5.0
        self.max_vel = 40.0

        # image properties
        self.img_count = 10
        self.dim_x = 15
        self.dim_y = 20
        self.noise_level = 1.0
        self.variance = self.noise_level**2
        self.p = psf(0.5)

        # create image set with single moving object
        self.imlist = []
        self.time_list = []
        for i in range(self.img_count):
            time = i / self.img_count
            self.time_list.append(time)
            im = layered_image(str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p)
            self.imlist.append(im)
        self.stack = image_stack(self.imlist)

        # Set up old_results object for analysis_utils.PostProcess
        self.num_curves = 4
        curve_num_times = 20
        # First 3 passing indices
        psi_curves = [np.array([1.0 + (x / 100) for x in range(curve_num_times)])
                      for _ in range(self.num_curves - 1)]
        phi_curves = [np.array([1.0 + (y / 100) for y in range(curve_num_times)])
                      for _ in range(self.num_curves - 1)]
        # Failing index
        # Failing index (generate a list of psi values such that the elements 2 and 14 are filtered
        # by sigmaG filtering.
        failing_psi = [0.0 + (z / 100) for z in range(curve_num_times)]
        failing_psi[14] = -100.0
        failing_psi[2] = 100.0
        psi_curves.append(np.array(failing_psi))
        phi_curves.append(np.array([1.0 for _ in range(curve_num_times)]))

        self.good_indices = [z for z in range(curve_num_times)]
        self.good_indices.remove(14)
        self.good_indices.remove(2)

        self.curve_result_set = ResultSet()
        self.curve_time_list = [i for i in range(curve_num_times)]
        for i in range(self.num_curves):
            row = ResultDataRow(trajectory(), self.curve_time_list)
            row.set_psi_phi(psi_curves[i], phi_curves[i])
            self.curve_result_set.append_result(row)

    def test_per_image_mask(self):
        kb_post_process = PostProcess(self.config, self.time_list)

        # Set each mask pixel in a row to one masking reason.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()
            for x in range(self.dim_x):
                msk.set_pixel(x, 3, 2**x)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        # Mask with the default keys.
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 3 and (x == 0 or x == 4 or x == 7 or x == 8 or x == 15):
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_mask_threshold(self):
        kb_post_process = PostProcess(self.config, self.time_list)

        # Set one science pixel per image above the threshold
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(2 + i, 8, 501.0)
            sci.set_pixel(1 + i, 9, 499.0)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_science(sci)
            self.stack.set_single_image(i, img)

        # With default threshold (None) nothing should be masked.
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    self.assertTrue(sci.pixel_has_data(x, y))

        # With a threshold of 500 one pixel per image should be masked.
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=500, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if x == 2 + i and y == 8:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_mask_grow(self):
        kb_post_process = PostProcess(self.config, self.time_list)

        # Set one science pixel per image above the threshold
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(2 + i, 8, 501.0)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_science(sci)
            self.stack.set_single_image(i, img)

        # With default threshold (None) nothing should be masked.
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=500, mask_grow=2)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    dist = abs(2 + i - x) + abs(y - 8)
                    self.assertEqual(sci.pixel_has_data(x, y), dist > 2)

    def test_global_mask(self):
        self.config["repeated_flag_keys"] = ["CR"]
        kb_post_process = PostProcess(self.config, self.time_list)

        # Set each mask pixel in a single row depending on the image number.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()
            for x in range(self.dim_x):
                if x >= i:
                    msk.set_pixel(x, 1, 8)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        # Apply the global mask with mask_num_images=5 and check that we mask the correct pixels.
        kb_post_process.apply_mask(self.stack, mask_num_images=5, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 1 and x >= 4:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

        # Apply the global mask with mask_num_images=3 and check that we mask 2 more pixels.
        kb_post_process.apply_mask(self.stack, mask_num_images=3, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 1 and x >= 2:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_apply_clipped_average_single_thread(self):
        # make sure apply_clipped_average works when num_cores == 1
        kb_post_process = PostProcess(self.config, self.curve_time_list)

        kb_post_process.apply_clipped_average(self.curve_result_set, {})

        # Check to ensure first three results have all indices passing and the
        # last index is missing two points.
        all_indices = [i for i in range(len(self.curve_time_list))]
        self.assertEqual(self.curve_result_set.results[0].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[1].valid_indices, all_indices)   
        self.assertEqual(self.curve_result_set.results[2].valid_indices, all_indices)     
        self.assertEqual(self.curve_result_set.results[3].valid_indices, self.good_indices)

    def test_apply_clipped_average_multi_thread(self):
        # make sure apply_clipped_average works when multithreading is enabled
        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config, self.time_list)

        kb_post_process.apply_clipped_average(self.curve_result_set, {})

        # Check to ensure first three results have all indices passing and the
        # last index is missing two points.
        all_indices = [i for i in range(len(self.curve_time_list))]
        self.assertEqual(self.curve_result_set.results[0].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[1].valid_indices, all_indices)   
        self.assertEqual(self.curve_result_set.results[2].valid_indices, all_indices)     
        self.assertEqual(self.curve_result_set.results[3].valid_indices, self.good_indices)

    def test_apply_clipped_sigmaG_single_thread(self):
        # make sure apply_clipped_sigmaG works when num_cores == 1
        kb_post_process = PostProcess(self.config, self.time_list)

        kb_post_process.apply_clipped_sigmaG(self.curve_result_set, {"sigmaG_filter_type": "lh"})

        # Check to ensure first three results have all indices passing and the
        # last index is missing two points.
        all_indices = [i for i in range(len(self.curve_time_list))]
        self.assertEqual(self.curve_result_set.results[0].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[1].valid_indices, all_indices)   
        self.assertEqual(self.curve_result_set.results[2].valid_indices, all_indices)     
        self.assertEqual(self.curve_result_set.results[3].valid_indices, self.good_indices)

    def test_apply_clipped_sigmaG_multi_thread(self):
        # make sure apply_clipped_sigmaG works when multithreading is enabled
        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config, self.time_list)

        kb_post_process.apply_clipped_sigmaG(self.curve_result_set, {"sigmaG_filter_type": "lh"})

        # Check to ensure first three results have all indices passing and the
        # last index is missing two points.
        all_indices = [i for i in range(len(self.curve_time_list))]
        self.assertEqual(self.curve_result_set.results[0].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[1].valid_indices, all_indices)   
        self.assertEqual(self.curve_result_set.results[2].valid_indices, all_indices)     
        self.assertEqual(self.curve_result_set.results[3].valid_indices, self.good_indices)

    def test_apply_stamp_filter(self):        
        # object properties
        self.object_flux = 250.0
        self.start_x = 4
        self.start_y = 3
        self.x_vel = 2.0
        self.y_vel = 1.0

        for i in range(self.img_count):
            time = i / self.img_count
            self.imlist[i].add_object(
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
            )

        stack = image_stack(self.imlist)
        search = stack_search(stack)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.img_count / 2),
        )

        mjds = np.array(stack.get_times())
        kb_post_process = PostProcess(self.config, mjds)

        keep = kb_post_process.load_and_filter_results(
            search,
            {},
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            filter_type="kalman",
            max_lh=self.config["max_lh"],
        )

        # Apply the stamp filter with default parameters.
        kb_post_process.apply_stamp_filter(keep, search)

        # Check that we get at least one result and those results have stamps.
        self.assertGreater(keep.num_results(), 0)
        for i in range(keep.num_results()):
            self.assertIsNotNone(keep.results[i].stamp)

    def test_apply_stamp_filter_2(self):
        # Also confirms that apply_stamp_filter works with a chunksize < number
        # of results.

        # object properties
        self.object_flux = 250.0
        self.start_x = 4
        self.start_y = 3
        self.x_vel = 2.0
        self.y_vel = 1.0

        for i in range(self.img_count):
            time = i / self.img_count
            self.imlist[i].add_object(
                self.start_x + time * self.x_vel,
                self.start_y + time * self.y_vel,
                self.object_flux,
            )

        stack = image_stack(self.imlist)
        search = stack_search(stack)

        # Create a first trajectory that matches perfectly.
        trj = trajectory()
        trj.x = self.start_x
        trj.y = self.start_y
        trj.x_v = self.x_vel
        trj.y_v = self.y_vel

        # Create a second trajectory that isn't any good.
        trj2 = trajectory()
        trj2.x = 1
        trj2.y = 1
        trj2.x_v = 0
        trj2.y_v = 0

        # Create a third trajectory that is close to good, but offset.
        trj3 = trajectory()
        trj3.x = trj.x + 2
        trj3.y = trj.y + 2
        trj3.x_v = trj.x_v
        trj3.y_v = trj.y_v

        # Create a fourth trajectory that is just close enough
        trj4 = trajectory()
        trj4.x = trj.x + 1
        trj4.y = trj.y + 1
        trj4.x_v = trj.x_v
        trj4.y_v = trj.y_v

        # Create the ResultSet.
        keep = ResultSet()
        keep.append_result(ResultDataRow(trj, self.time_list))
        keep.append_result(ResultDataRow(trj2, self.time_list))
        keep.append_result(ResultDataRow(trj3, self.time_list))
        keep.append_result(ResultDataRow(trj4, self.time_list))

        # Create the post processing object.
        kb_post_process = PostProcess(self.config, self.time_list)
        keep2 = kb_post_process.apply_stamp_filter(
            keep,
            search,
            center_thresh=0.03,
            peak_offset=[1.5, 1.5],
            mom_lims=[35.5, 35.5, 1.0, 1.0, 1.0],
            chunk_size=1,
            stamp_type="cpp_mean",
            stamp_radius=5,
        )

        # The check that the correct indices and number of stamps are saved.
        self.assertEqual(keep.num_results(), 2)
        self.assertEqual(keep.results[0].trajectory.x, self.start_x)
        self.assertEqual(keep.results[1].trajectory.x, self.start_x + 1)

    def test_get_coadded_stamps(self):
        # object properties
        self.object_flux = 250.0
        self.start_x = 4
        self.start_y = 3
        self.x_vel = 2.0
        self.y_vel = 1.0

        for i in range(self.img_count):
            time = i / self.img_count
            self.imlist[i].add_object(
                self.start_x + time * self.x_vel,
                self.start_y + time * self.y_vel,
                self.object_flux,
            )

        stack = image_stack(self.imlist)
        search = stack_search(stack)

        # Create a first trajectory that matches perfectly.
        trj = trajectory()
        trj.x = self.start_x
        trj.y = self.start_y
        trj.x_v = self.x_vel
        trj.y_v = self.y_vel

        # Create the ResultSet.
        keep = ResultSet()
        for i in range(5):
            keep.append_result(ResultDataRow(trj, self.time_list))

        # Mark a few of the results as invalid for trajectories 2 and 3.
        keep.results[2].filter_indices([2, 3, 4, 7, 8, 9])
        keep.results[3].filter_indices([0, 1, 5, 6])

        # Create the post processing object.
        result_idx = [1, 3, 4]
        kb_post_process = PostProcess(self.config, self.time_list)
        stamps = kb_post_process.get_coadd_stamps(result_idx, search, keep, 3, "cpp_mean")

        # Check that we only get three stamps back.
        self.assertEqual(len(stamps), 3)

        # Check that we are using the correct trajectories and lc_indices.
        for i, idx in enumerate(result_idx):
            res_trj = trj_result(trj, self.img_count, keep.results[idx].valid_indices)
            stamp_idv = search.mean_sci_stamp(res_trj, 3, False)
            stamp_batch = stamps[i]
            for x in range(7):
                for y in range(7):
                    self.assertAlmostEqual(stamp_idv.get_pixel(x, y), stamp_batch[y][x], delta=1e-5)

    def test_clustering(self):
        cluster_params = {}
        cluster_params["x_size"] = self.dim_x
        cluster_params["y_size"] = self.dim_y
        cluster_params["vel_lims"] = [self.min_vel, self.max_vel]
        cluster_params["ang_lims"] = [self.min_angle, self.max_angle]
        cluster_params["mjd"] = np.array(self.stack.get_times())

        trjs = [
            self._make_trajectory(10, 11, 1, 2, 100.0),
            self._make_trajectory(10, 11, 10, 20, 100.0),
            self._make_trajectory(40, 5, -1, 2, 100.0),
            self._make_trajectory(5, 0, 1, 2, 100.0),
            self._make_trajectory(5, 1, 1, 2, 100.0),
        ]

        # Try clustering with positions, velocities, and angles.
        self.config["cluster_type"] = "all"
        self.config["eps"] = 0.1
        kb_post_process = PostProcess(self.config, self.time_list)

        results = ResultSet()
        for t in trjs:
            results.append_result(ResultDataRow(t, self.time_list))
        results_dict = kb_post_process.apply_clustering(results, cluster_params)
        self.assertEqual(results.num_results(), 4)
 
        # Try clustering with only positions.
        self.config["cluster_type"] = "position"
        kb_post_process = PostProcess(self.config, self.time_list)
        keep = kb_post_process.gen_results_dict()

        results2 = ResultSet()
        for t in trjs:
            results2.append_result(ResultDataRow(t, self.time_list))
        results_dict = kb_post_process.apply_clustering(results2, cluster_params)
        self.assertEqual(results2.num_results(), 3)


if __name__ == "__main__":
    unittest.main()
