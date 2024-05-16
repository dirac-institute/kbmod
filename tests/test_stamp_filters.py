import numpy as np
import unittest

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import add_fake_object, create_fake_times, FakeDataSet
from kbmod.filters.stamp_filters import *
from kbmod.results import Results
from kbmod.search import *


class test_stamp_filters(unittest.TestCase):
    def setUp(self):
        # Create a fake data set to use in the tests.
        self.image_count = 10
        self.fake_times = create_fake_times(self.image_count, 57130.2, 1, 0.01, 1)
        self.ds = FakeDataSet(
            25,  # width
            35,  # height
            self.fake_times,  # time stamps
            1.0,  # noise level
            0.5,  # psf value
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        self.trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        self.ds.insert_object(self.trj)

    def test_extract_search_parameters_from_config(self):
        config_dict = {
            "center_thresh": 0.05,
            "do_stamp_filter": True,
            "mom_lims": [50.0, 51.0, 1.0, 2.0, 3.0],
            "peak_offset": [1.5, 3.5],
            "stamp_type": "median",
            "stamp_radius": 7,
        }
        config = SearchConfiguration.from_dict(config_dict)

        params = extract_search_parameters_from_config(config)
        self.assertEqual(params.radius, 7)
        self.assertEqual(params.stamp_type, StampType.STAMP_MEDIAN)
        self.assertEqual(params.do_filtering, True)
        self.assertAlmostEqual(params.center_thresh, 0.05)
        self.assertAlmostEqual(params.peak_offset_x, 1.5)
        self.assertAlmostEqual(params.peak_offset_y, 3.5)
        self.assertAlmostEqual(params.m20_limit, 50.0)
        self.assertAlmostEqual(params.m02_limit, 51.0)
        self.assertAlmostEqual(params.m11_limit, 1.0)
        self.assertAlmostEqual(params.m10_limit, 2.0)
        self.assertAlmostEqual(params.m01_limit, 3.0)

        # Test bad configurations
        config.set("stamp_radius", -1)
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("stamp_radius", 7)

        config.set("stamp_type", "broken")
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("stamp_type", "median")

        config.set("peak_offset", [50.0])
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("peak_offset", [1.5, 3.5])

        config.set("mom_lims", [50.0, 51.0, 1.0, 3.0])
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("mom_lims", [50.0, 51.0, 1.0, 2.0, 3.0])

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_get_coadds_and_filter_results(self):
        # Create trajectories to test: 0) known good, 1) completely wrong
        # 2) close to good, but offset], and 3) just close enough.
        trj_list = [
            self.trj,
            Trajectory(1, 1, 0.0, 0.0),
            Trajectory(self.trj.x + 2, self.trj.y + 2, self.trj.vx, self.trj.vy),
            Trajectory(self.trj.x + 1, self.trj.y + 1, self.trj.vx, self.trj.vy),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("stamp" in keep.colnames)

        # Create the stamp parameters we need.
        config_dict = {
            "center_thresh": 0.03,
            "do_stamp_filter": True,
            "mom_lims": [35.5, 35.5, 1.0, 1.0, 1.0],
            "peak_offset": [1.5, 1.5],
            "stamp_type": "cpp_mean",
            "stamp_radius": 5,
        }
        config = SearchConfiguration.from_dict(config_dict)

        # Do the filtering.
        get_coadds_and_filter_results(keep, self.ds.stack, config, chunk_size=2)

        # The check that the correct indices and number of stamps are saved.
        self.assertTrue("stamp" in keep.colnames)
        self.assertEqual(len(keep), 2)
        self.assertEqual(keep["x"][0], self.trj.x)
        self.assertEqual(keep["x"][1], self.trj.x + 1)
        self.assertEqual(keep["stamp"][0].shape, (11, 11))
        self.assertEqual(keep["stamp"][1].shape, (11, 11))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_get_coadds_and_filter_with_invalid(self):
        valid1 = [True] * self.image_count
        valid2 = [True] * self.image_count
        # Completely mess up some of the images.
        for i in [1, 3, 6, 7, 9]:
            self.ds.stack.get_single_image(i).get_science().set_all(1000.0)
            valid2[i] = False

        # Create the Results with nearly identical trajectories,
        # but different valid observations
        trj2 = Trajectory(self.trj.x, self.trj.y, self.trj.vx, self.trj.vy + 0.001, flux=250.0)
        keep = Results.from_trajectories([self.trj, trj2])
        keep.update_obs_valid(np.array([valid1, valid2]))

        # Create the stamp parameters we need.
        config_dict = {
            "center_thresh": 0.03,
            "do_stamp_filter": True,
            "mom_lims": [35.5, 35.5, 1.0, 1.0, 1.0],
            "peak_offset": [1.5, 1.5],
            "stamp_type": "cpp_mean",
            "stamp_radius": 5,
        }
        config = SearchConfiguration.from_dict(config_dict)

        # Do the filtering.
        get_coadds_and_filter_results(keep, self.ds.stack, config, chunk_size=2)

        # The check that the correct indices and number of stamps are saved.
        self.assertTrue("stamp" in keep.colnames)
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep["vx"][0], trj2.vx)
        self.assertEqual(keep["vy"][0], trj2.vy)

        # Test with empty results.
        keep2 = Results.from_trajectories([])
        get_coadds_and_filter_results(keep2, self.ds.stack, config, chunk_size=1000)
        self.assertTrue("stamp" in keep2.colnames)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_append_coadds(self):
        # Create trajectories to test: 0) known good, 1) completely wrong
        # 2) close to good, but offset], 3) just close enough, and
        # 4) another wrong one.
        trj_list = [
            self.trj,
            Trajectory(1, 1, 0.0, 0.0),
            Trajectory(self.trj.x + 2, self.trj.y + 2, self.trj.vx, self.trj.vy),
            Trajectory(self.trj.x + 1, self.trj.y + 1, self.trj.vx, self.trj.vy),
            Trajectory(10, 3, 0.1, -0.1),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertFalse("coadd_mean" in keep.colnames)
        self.assertFalse("coadd_median" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Adding nothing does nothing.
        append_coadds(keep, self.ds.stack, [], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertFalse("coadd_mean" in keep.colnames)
        self.assertFalse("coadd_median" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Adding "mean" and "median" does only those.
        append_coadds(keep, self.ds.stack, ["median", "mean"], 3)
        self.assertFalse("coadd_sum" in keep.colnames)
        self.assertTrue("coadd_mean" in keep.colnames)
        self.assertTrue("coadd_median" in keep.colnames)
        self.assertFalse("stamp" in keep.colnames)

        # Check that all coadds are generated without filtering.
        for i in range(len(trj_list)):
            self.assertEqual(keep["coadd_mean"][i].shape, (7, 7))
            self.assertEqual(keep["coadd_median"][i].shape, (7, 7))

    def test_append_all_stamps(self):
        # Make a few results with different trajectories.
        trj_list = [
            Trajectory(8, 7, 2.0, 1.0),
            Trajectory(10, 22, -2.0, -1.0),
            Trajectory(8, 7, -2.0, -1.0),
        ]
        keep = Results.from_trajectories(trj_list)
        self.assertFalse("all_stamps" in keep.colnames)

        append_all_stamps(keep, self.ds.stack, 5)
        self.assertTrue("all_stamps" in keep.colnames)
        for i in range(len(keep)):
            stamps_array = keep["all_stamps"][i]
            self.assertEqual(stamps_array.shape[0], self.image_count)
            self.assertEqual(stamps_array.shape[1], 11)
            self.assertEqual(stamps_array.shape[2], 11)

        # Check that everything works if the results are empty.
        keep2 = Results.from_trajectories([])
        append_all_stamps(keep2, self.ds.stack, 5)
        self.assertTrue("all_stamps" in keep2.colnames)


if __name__ == "__main__":
    unittest.main()
