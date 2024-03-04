import unittest

from kbmod.analysis_utils import *
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.result_list import *
from kbmod.search import *
from kbmod.trajectory_utils import make_trajectory

from utils.utils_for_tests import get_absolute_data_path


class test_analysis_utils(unittest.TestCase):
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
            "encode_num_bytes": -1,
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
        self.dim_x = 100
        self.dim_y = 100

        # Create fake images.
        self.time_list = [i / self.img_count for i in range(self.img_count)]
        self.fake_ds = FakeDataSet(
            self.dim_x,
            self.dim_y,
            self.time_list,
            noise_level=1.0,
            psf_val=0.5,
            use_seed=True,
        )
        self.stack = self.fake_ds.stack

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

    def test_load_and_filter_results_lh(self):
        # Create fake result trajectories with given initial likelihoods. The 1st is
        # filtered by max likelihood. The 4th and 5th are filtered by min likelihood.
        trjs = [
            make_trajectory(20, 20, 0, 0, 500.0, 9000.0, self.img_count),
            make_trajectory(30, 30, 0, 0, 100.0, 100.0, self.img_count),
            make_trajectory(40, 40, 0, 0, 50.0, 50.0, self.img_count),
            make_trajectory(50, 50, 0, 0, 1.0, 2.0, self.img_count),
            make_trajectory(60, 60, 0, 0, 1.0, 1.0, self.img_count),
        ]

        # Add a fake object to the fake images.
        for trj in trjs:
            self.fake_ds.insert_object(trj)

        # Create the stack search and insert the fake results.
        search = StackSearch(self.fake_ds.stack)
        search.set_results(trjs)

        # Do the filtering.
        self.config["num_obs"] = 5
        kb_post_process = PostProcess(self.config, self.time_list)

        results = kb_post_process.load_and_filter_results(
            search,
            10.0,  # min likelihood
            chunk_size=500000,
            max_lh=1000.0,
        )

        # Only two of the middle results should pass the filtering.
        self.assertEqual(results.num_results(), 2)
        self.assertEqual(results.results[0].trajectory.y, 30)
        self.assertEqual(results.results[1].trajectory.y, 40)


if __name__ == "__main__":
    unittest.main()
