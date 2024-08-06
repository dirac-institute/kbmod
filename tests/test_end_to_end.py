# import math
# import numpy as np
# import os
# import tempfile
# import pytest
#
# from kbmod.fake_data.fake_data_creator import *
# from kbmod.run_search import *
# from kbmod.search import *
# from kbmod.wcs_utils import make_fake_wcs
# from kbmod.work_unit import WorkUnit

from utils.utils_for_tests import get_absolute_demo_data_path


####
import unittest

from kbmod import ImageCollection
from kbmod.run_search import SearchRunner
from kbmod.configuration import SearchConfiguration
import kbmod.mocking as kbmock


class TestEmptySearch(unittest.TestCase):
    def setUp(self):
        self.factory = kbmock.EmptyFits()

    def test_empty(self):
        hduls = self.factory.mock(n=10)

        # create the most permissive search configs you can come up with
        # all values in these images are zeros, we should not be returning
        # anything
        config = SearchConfiguration.from_dict(
            {
                "average_angle": 0.0,
                "v_arr": [10, 20, 10],
                "lh_level": 0.1,
                "num_obs": 1,
                "do_mask": False,
                "do_clustering": True,
                "do_stamp_filter": False,
            }
        )

        ic = ImageCollection.fromTargets(hduls, force="TestDataStd")
        wu = ic.toWorkUnit(search_config=config)
        results = SearchRunner().run_search_from_work_unit(wu)
        self.assertTrue(len(results) == 0)

    def test_static_objects(self):
        src_cat = kbmock.SourceCatalog.from_defaults(seed=100)
        factory = kbmock.SimpleFits(src_cat=src_cat)
        hduls = factory.mock(10)

        ic = ImageCollection.fromTargets(hduls, force="TestDataStd")
        wu = ic.toWorkUnit(search_config=SearchConfiguration())
        results = SearchRunner().run_search_from_work_unit(wu)
        self.assertTrue(len(results) == 0)


class TestLinearSearch(unittest.TestCase):
    def setUp(self):
        # Set up shared search values
        self.n_imgs = 10
        self.repeat_n_times = 10
        self.shape = (500, 500)
        self.start_pos = (10, 50)
        self.vxs = [10, 30]
        self.vys = [10, 30]

        # Set up configs for mocking and search
        # These don't change from test to test
        self.param_ranges = {
            "amplitude": [100, 100],
            "x_mean": self.start_pos,
            "y_mean": self.start_pos,
            "x_stddev": [2.0, 2.0],
            "y_stddev": [2.0, 2.0],
            "vx": self.vxs,
            "vy": self.vys,
        }

        self.config = SearchConfiguration.from_dict(
            {
                "generator_config": {
                    "name": "VelocityGridSearch",
                    "min_vx": self.vxs[0],
                    "max_vx": self.vxs[1],
                    "min_vy": self.vys[0],
                    "max_vy": self.vys[1],
                    "vx_steps": 50,
                    "vy_steps": 50,
                },
                "num_obs": 10,
                "do_mask": False,
                "do_clustering": True,
                "do_stamp_filter": False,
            }
        )

    def test_search(self):
        # Mock the data and repeat tests. The random catalog
        # creation guarantees a diverse set of changing test values
        for i in range(self.repeat_n_times):
            with self.subTest(n=i):
                obj_cat = kbmock.ObjectCatalog.from_defaults(self.param_ranges, n=1)
                factory = kbmock.SimpleFits(shape=self.shape, step_mjd=1, obj_cat=obj_cat)
                hduls = factory.mock(n=self.n_imgs)

                ic = ImageCollection.fromTargets(hduls, force="TestDataStd")
                wu = ic.toWorkUnit(search_config=self.config)
                results = SearchRunner().run_search_from_work_unit(wu)

                # Run tests
                self.assertGreaterEqual(len(results), 1)
                for res in results:
                    diff = abs(obj_cat.table["y_mean"] - res["y"])
                    obj = obj_cat.table[diff == diff.min()]
                    self.assertLessEqual(abs(obj["x_mean"] - res["x"]), 5)
                    self.assertLessEqual(abs(obj["y_mean"] - res["y"]), 5)
                    self.assertLessEqual(abs(obj["vx"] - res["vx"]), 5)
                    self.assertLessEqual(abs(obj["vy"] - res["vy"]), 5)


####


# this is the first test to actually test things like get_all_stamps from
# analysis utils. For now stamps have to be RawImages (because methods like
# interpolate and convolve are defined to work on RawImage and not as funciton)
# so it makes sense to duplicate all this functionality to return np arrays
# (instead of RawImages), but hopefully we can deduplicate all this by making
# these operations into functions and calling on the .image attribute
# apply_stamp_filter for example is literal copy of the C++ code in RawImage?
# class test_end_to_end(pytest.TestCase):
#    def setUp(self):
#        # Define the path for the data.
#        im_filepath = get_absolute_demo_data_path("demo")
#
#        # The demo data has an object moving at x_v=10 px/day
#        # and y_v = 0 px/day. So we search velocities [0, 20]
#        # and angles [-0.5, 0.5].
#        v_arr = [0, 20, 21]
#        ang_arr = [0.5, 0.5, 11]
#
#        self.input_parameters = {
#            # Required
#            "im_filepath": im_filepath,
#            "res_filepath": None,
#            "time_file": None,
#            "output_suffix": "DEMO",
#            "v_arr": v_arr,
#            "ang_arr": ang_arr,
#            # Important
#            "num_obs": 7,
#            "do_mask": True,
#            "lh_level": 10.0,
#            "gpu_filter": True,
#            # Fine tuning
#            "sigmaG_lims": [15, 60],
#            "mom_lims": [37.5, 37.5, 1.5, 1.0, 1.0],
#            "peak_offset": [3.0, 3.0],
#            "chunk_size": 1000000,
#            "stamp_type": "cpp_median",
#            "eps": 0.03,
#            "clip_negative": True,
#            "mask_num_images": 10,
#            "cluster_type": "position",
#            # Override the ecliptic angle for the demo data since we
#            # know the true angle in pixel space.
#            "average_angle": 0.0,
#            "save_all_stamps": True,
#        }
#
#    @pytest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
#    def test_demo_defaults(self):
#        rs = SearchRunner()
#        keep = rs.run_search_from_config(self.input_parameters)
#        self.assertGreaterEqual(len(keep), 1)
#        self.assertEqual(keep["stamp"][0].shape, (21, 21))
#
#    @pytest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
#    def test_demo_config_file(self):
#        im_filepath = get_absolute_demo_data_path("demo")
#        config_file = get_absolute_demo_data_path("demo_config.yml")
#        rs = SearchRunner()
#        keep = rs.run_search_from_file(
#            config_file,
#            overrides={"im_filepath": im_filepath},
#        )
#        self.assertGreaterEqual(len(keep), 1)
#        self.assertEqual(keep["stamp"][0].shape, (21, 21))
#
#    @pytest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
#    def test_demo_stamp_size(self):
#        self.input_parameters["stamp_radius"] = 15
#        self.input_parameters["mom_lims"] = [80.0, 80.0, 50.0, 20.0, 20.0]
#
#        rs = SearchRunner()
#        keep = rs.run_search_from_config(self.input_parameters)
#        self.assertGreaterEqual(len(keep), 1)
#
#        self.assertIsNotNone(keep["stamp"][0])
#        self.assertEqual(keep["stamp"][0].shape, (31, 31))
#
#        self.assertIsNotNone(keep["all_stamps"][0])
#        for s in keep["all_stamps"][0]:
#            self.assertEqual(s.shape, (31, 31))
#
#    @pytest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
#    def test_e2e_work_unit(self):
#        num_images = 10
#
#        # Create a fake data set with a single bright fake object and all
#        # the observations on a single day.
#        fake_times = create_fake_times(num_images, 57130.2, 10, 0.01, 1)
#        ds = FakeDataSet(128, 128, fake_times, use_seed=True)
#        trj = Trajectory(x=50, y=60, vx=5.0, vy=0.0, flux=500.0)
#        ds.insert_object(trj)
#
#        # Set the configuration to pick up the fake object.
#        config = SearchConfiguration()
#        config.set("ang_arr", [math.pi, math.pi, 16])
#        config.set("v_arr", [0, 10.0, 20])
#
#        fake_wcs = make_fake_wcs(10.0, 10.0, 128, 128)
#        work = WorkUnit(im_stack=ds.stack, config=config, wcs=fake_wcs)
#
#        with tempfile.TemporaryDirectory() as dir_name:
#            file_path = os.path.join(dir_name, "test_workunit.fits")
#            work.to_fits(file_path)
#
#            rs = SearchRunner()
#            keep = rs.run_search_from_file(file_path)
#            self.assertGreaterEqual(len(keep), 1)


if __name__ == "__main__":
    unittest.main()
