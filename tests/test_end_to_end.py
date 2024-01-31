import math
import numpy as np
import os
import tempfile
import unittest

from kbmod.fake_data_creator import *
from kbmod.run_search import *
from kbmod.search import *
from kbmod.work_unit import WorkUnit

# from .utils_for_tests import get_absolute_demo_data_path
# import utils_for_tests
from utils.utils_for_tests import get_absolute_demo_data_path


# this is the first test to actually test things like get_all_stamps from
# analysis utils. For now stamps have to be RawImages (because methods like
# interpolate and convolve are defined to work on RawImage and not as funciton)
# so it makes sense to duplicate all this functionality to return np arrays
# (instead of RawImages), but hopefully we can deduplicate all this by making
# these operations into functions and calling on the .image attribute
# apply_stamp_filter for example is literal copy of the C++ code in RawImage?
class test_end_to_end(unittest.TestCase):
    def setUp(self):
        # Define the path for the data.
        im_filepath = get_absolute_demo_data_path("demo")

        # The demo data has an object moving at x_v=10 px/day
        # and y_v = 0 px/day. So we search velocities [0, 20]
        # and angles [-0.5, 0.5].
        v_arr = [0, 20, 21]
        ang_arr = [0.5, 0.5, 11]

        self.input_parameters = {
            # Required
            "im_filepath": im_filepath,
            "res_filepath": None,
            "time_file": None,
            "output_suffix": "DEMO",
            "v_arr": v_arr,
            "ang_arr": ang_arr,
            # Important
            "num_obs": 7,
            "do_mask": True,
            "lh_level": 10.0,
            "gpu_filter": True,
            # Fine tuning
            "sigmaG_lims": [15, 60],
            "mom_lims": [37.5, 37.5, 1.5, 1.0, 1.0],
            "peak_offset": [3.0, 3.0],
            "chunk_size": 1000000,
            "stamp_type": "cpp_median",
            "eps": 0.03,
            "clip_negative": True,
            "mask_num_images": 10,
            "cluster_type": "position",
            # Override the ecliptic angle for the demo data since we
            # know the true angle in pixel space.
            "average_angle": 0.0,
        }

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_demo_defaults(self):
        rs = SearchRunner()
        keep = rs.run_search_from_config(self.input_parameters)
        self.assertGreaterEqual(keep.num_results(), 1)
        self.assertEqual(keep.results[0].stamp.shape, (21, 21))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_demo_config_file(self):
        im_filepath = get_absolute_demo_data_path("demo")
        config_file = get_absolute_demo_data_path("demo_config.yml")
        rs = SearchRunner()
        keep = rs.run_search_from_file(
            config_file,
            overrides={"im_filepath": im_filepath},
        )
        self.assertGreaterEqual(keep.num_results(), 1)
        self.assertEqual(keep.results[0].stamp.shape, (21, 21))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_demo_stamp_size(self):
        self.input_parameters["stamp_radius"] = 15
        self.input_parameters["mom_lims"] = [80.0, 80.0, 50.0, 20.0, 20.0]

        rs = SearchRunner()
        keep = rs.run_search_from_config(self.input_parameters)
        self.assertGreaterEqual(keep.num_results(), 1)

        self.assertIsNotNone(keep.results[0].stamp)
        self.assertEqual(keep.results[0].stamp.shape, (31, 31))

        self.assertIsNotNone(keep.results[0].all_stamps)
        for s in keep.results[0].all_stamps:
            self.assertEqual(s.shape, (31, 31))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_e2e_work_unit(self):
        num_images = 10

        # Create a fake data set with a single bright fake object.
        ds = FakeDataSet(128, 128, num_images, obs_per_day=10, use_seed=True)
        trj = Trajectory()
        trj.x = 50
        trj.y = 60
        trj.vx = 5.0
        trj.vy = 0.0
        trj.flux = 500.0
        ds.insert_object(trj)

        # Set the configuration to pick up the fake object.
        config = SearchConfiguration()
        config.set("ang_arr", [math.pi, math.pi, 16])
        config.set("v_arr", [0, 10.0, 20])

        work = WorkUnit(im_stack=ds.stack, config=config)

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test_workunit.fits")
            work.to_fits(file_path)

            rs = SearchRunner()
            keep = rs.run_search_from_file(file_path)
            self.assertGreaterEqual(keep.num_results(), 1)


if __name__ == "__main__":
    unittest.main()
