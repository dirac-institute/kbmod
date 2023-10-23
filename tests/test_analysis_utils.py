import unittest

from kbmod.analysis_utils import *
from kbmod.fake_data_creator import add_fake_object
from kbmod.result_list import *
from kbmod.search import *

from utils.utils_for_tests import get_absolute_data_path


class test_analysis_utils(unittest.TestCase):
    def _make_trajectory(self, x0, y0, xv, yv, lh):
        t = Trajectory()
        t.x = x0
        t.y = y0
        t.vx = xv
        t.vy = yv
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
            "file_format": "{0:06d}.fits",
            "sigmaG_lims": [25, 75],
            "chunk_size": 500000,
            "max_lh": 1000.0,
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
            "cluster_type": "all",
            "cluster_function": "DBSCAN",
            "mask_bits_dict": self.default_mask_bits_dict,
            "flag_keys": self.default_flag_keys,
            "repeated_flag_keys": self.default_repeated_flag_keys,
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
        self.p = PSF(0.5)

        # create image set with single moving object
        self.imlist = []
        self.time_list = []
        for i in range(self.img_count):
            time = i / self.img_count
            self.time_list.append(time)
            im = LayeredImage(
                str(i), self.dim_y, self.dim_x, self.noise_level, self.variance, time, self.p, i
            )
            self.imlist.append(im)
        self.stack = ImageStack(self.imlist)

        # Set up old_results object for analysis_utils.PostProcess
        self.num_curves = 4
        curve_num_times = 20
        # First 3 passing indices
        get_psi_curves = [
            np.array([1.0 + (x / 100) for x in range(curve_num_times)]) for _ in range(self.num_curves - 1)
        ]
        get_phi_curves = [
            np.array([1.0 + (y / 100) for y in range(curve_num_times)]) for _ in range(self.num_curves - 1)
        ]
        # Failing index
        # Failing index (generate a list of psi values such that the elements 2 and 14 are filtered
        # by sigmaG filtering.
        failing_psi = [0.0 + (z / 100) for z in range(curve_num_times)]
        failing_psi[14] = -100.0
        failing_psi[2] = 100.0
        get_psi_curves.append(np.array(failing_psi))
        get_phi_curves.append(np.array([1.0 for _ in range(curve_num_times)]))

        self.good_indices = [z for z in range(curve_num_times)]
        self.good_indices.remove(14)
        self.good_indices.remove(2)

        self.curve_time_list = [i for i in range(curve_num_times)]
        self.curve_result_set = ResultList(self.curve_time_list)
        for i in range(self.num_curves):
            row = ResultRow(Trajectory(), curve_num_times)
            row.set_psi_phi(get_psi_curves[i], get_phi_curves[i])
            self.curve_result_set.append_result(row)

    def test_apply_clipped_sigmaG_single_thread(self):
        # make sure apply_clipped_sigmaG works when num_cores == 1
        kb_post_process = PostProcess(self.config, self.time_list)

        kb_post_process.apply_clipped_sigmaG(self.curve_result_set)

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

        kb_post_process.apply_clipped_sigmaG(self.curve_result_set)

        # Check to ensure first three results have all indices passing and the
        # last index is missing two points.
        all_indices = [i for i in range(len(self.curve_time_list))]
        self.assertEqual(self.curve_result_set.results[0].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[1].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[2].valid_indices, all_indices)
        self.assertEqual(self.curve_result_set.results[3].valid_indices, self.good_indices)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_apply_stamp_filter(self):
        # object properties
        self.object_flux = 250.0
        self.start_x = 4
        self.start_y = 3
        self.vxel = 2.0
        self.vyel = 1.0

        for i in range(self.img_count):
            time = i / self.img_count
            add_fake_object(
                self.imlist[i],
                self.start_y + time * self.vyel + 0.5,
                self.start_x + time * self.vxel + 0.5,
                self.object_flux,
                self.p,
            )

        stack = ImageStack(self.imlist)
        search = StackSearch(stack)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.img_count / 2),
        )

        zeroed_times = np.array(stack.build_zeroed_times())
        kb_post_process = PostProcess(self.config, zeroed_times)

        keep = kb_post_process.load_and_filter_results(
            search,
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
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
        self.vxel = 2.0
        self.vyel = 1.0

        for i in range(self.img_count):
            time = i / self.img_count
            add_fake_object(
                self.imlist[i],
                self.start_y + time * self.vyel,
                self.start_x + time * self.vxel,
                self.object_flux,
                self.p,
            )

        stack = ImageStack(self.imlist)
        search = StackSearch(stack)

        # Create a first Trajectory that matches perfectly.
        trj = Trajectory()
        trj.x = self.start_x
        trj.y = self.start_y
        trj.vx = self.vxel
        trj.vy = self.vyel

        # Create a second Trajectory that isn't any good.
        trj2 = Trajectory()
        trj2.x = 1
        trj2.y = 1
        trj2.vx = 0
        trj2.vy = 0

        # Create a third Trajectory that is close to good, but offset.
        trj3 = Trajectory()
        trj3.x = trj.x + 2
        trj3.y = trj.y + 2
        trj3.vx = trj.vx
        trj3.vy = trj.vy

        # Create a fourth Trajectory that is just close enough
        trj4 = Trajectory()
        trj4.x = trj.x + 1
        trj4.y = trj.y + 1
        trj4.vx = trj.vx
        trj4.vy = trj.vy

        # Create the ResultList.
        keep = ResultList(self.time_list)
        keep.append_result(ResultRow(trj, self.img_count))
        keep.append_result(ResultRow(trj2, self.img_count))
        keep.append_result(ResultRow(trj3, self.img_count))
        keep.append_result(ResultRow(trj4, self.img_count))

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

    def test_clustering(self):
        cluster_params = {}
        cluster_params["x_size"] = self.dim_x
        cluster_params["y_size"] = self.dim_y
        cluster_params["vel_lims"] = [self.min_vel, self.max_vel]
        cluster_params["ang_lims"] = [self.min_angle, self.max_angle]
        cluster_params["mjd"] = np.array(self.stack.build_zeroed_times())

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

        results = ResultList(self.time_list)
        for t in trjs:
            results.append_result(ResultRow(t, self.img_count))
        results_dict = kb_post_process.apply_clustering(results, cluster_params)
        self.assertEqual(results.num_results(), 4)

        # Try clustering with only positions.
        self.config["cluster_type"] = "position"
        kb_post_process = PostProcess(self.config, self.time_list)

        results2 = ResultList(self.time_list)
        for t in trjs:
            results2.append_result(ResultRow(t, self.img_count))
        results_dict = kb_post_process.apply_clustering(results2, cluster_params)
        self.assertEqual(results2.num_results(), 3)

    def test_load_and_filter_results_lh(self):
        # Create fake result trajectories with given initial likelihoods.
        trjs = [
            self._make_trajectory(20, 20, 0, 0, 9000.0),  # Filtered by max likelihood
            self._make_trajectory(30, 30, 0, 0, 100.0),
            self._make_trajectory(40, 40, 0, 0, 50.0),
            self._make_trajectory(50, 50, 0, 0, 2.0),  # Filtered by min likelihood
            self._make_trajectory(60, 60, 0, 0, 1.0),  # Filtered by min likelihood
        ]
        fluxes = [500.0, 100.0, 50.0, 1.0, 0.1]

        # Create fake images with the objects in them.
        imlist = []
        for i in range(self.img_count):
            t = self.time_list[i]
            im = LayeredImage(str(i), 100, 100, self.noise_level, self.variance, t, self.p, i)

            # Add the objects.
            for j, trj in enumerate(trjs):
                add_fake_object(im, trj.y, trj.x, fluxes[j], self.p)

            # Append the image.
            imlist.append(im)

        # Create the stack search and insert the fake results.
        search = StackSearch(ImageStack(imlist))
        search.set_results(trjs)

        # Do the filtering.
        kb_post_process = PostProcess(self.config, self.time_list)
        results = kb_post_process.load_and_filter_results(
            search,
            10.0,  # min likelihood
            chunk_size=500000,
            max_lh=1000.0,
        )

        # Only the middle two results should pass the filtering.
        self.assertEqual(results.num_results(), 2)
        self.assertEqual(results.results[0].trajectory.y, 30)
        self.assertEqual(results.results[1].trajectory.y, 40)


if __name__ == "__main__":
    unittest.main()
