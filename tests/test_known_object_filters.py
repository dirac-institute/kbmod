import random
import unittest

import numpy as np
from astropy.table import Table

from kbmod.fake_data.fake_data_creator import FakeDataSet, create_fake_times
from kbmod.filters.known_object_filters import KnownObjsMatcher
from kbmod.results import Results
from kbmod.search import *
from kbmod.trajectory_utils import trajectory_predict_skypos
from kbmod.wcs_utils import make_fake_wcs


class TestKnownObjFilters(unittest.TestCase):
    def setUp(self):
        self.seed = 500  # Seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.filter_params = {
            "filter_type": "test_matches",
            "known_obj_thresh": 0.5,
            "known_obj_sep_thresh": 1.0,
            "known_obj_sep_time_thresh_s": 600,
            "known_obj_match_obs_ratio": 0.5,
            "known_obj_match_min_obs": 5,
        }

        num_images = 25
        self.obstimes = np.array(create_fake_times(num_images))
        # Create a fake dataset with 15 x 10 images.
        ds = FakeDataSet(15, 10, self.obstimes, use_seed=True)
        self.wcs = make_fake_wcs(10.0, 15.0, 15, 10)
        ds.set_wcs(self.wcs)

        num_results = 10
        # Randomly generate a Trajectory for each result, generating random x, y, vx, and vy
        for i in range(num_results):
            ds.insert_random_object(self.seed)
        self.res = Results.from_trajectories(ds.trajectories, track_filtered=True)
        self.assertEqual(len(ds.trajectories), num_results)

        # Generate which observations are valid observations for each result
        self.obs_valid = np.full((num_results, num_images), True)
        for i in range(num_results):
            invalid_obs = np.random.choice(num_images, 5, replace=False)
            self.obs_valid[i][invalid_obs] = False
        self.res.update_obs_valid(self.obs_valid)
        assert set(self.res.table.columns) == set(
            ["x", "y", "vx", "vy", "likelihood", "flux", "obs_count", "obs_valid"]
        )

        # Use the results' trajectories to generate a set of known objects that we can use to test the filter
        # Now we want to create a data set of known objects that interset our generated results in various
        # ways.
        self.known_objs = Table({"Name": np.empty(0, dtype=str), "RA": [], "DEC": [], "mjd_mid": []})

        # Case 1: Near in space (<1") and near in time (>1 s) and near in time to result 1
        self.generate_known_obj_from_result(
            self.known_objs,
            1,  # Base off result 1
            self.obstimes,  # Use all possible obstimes
            "spatial_close_time_close_1",
            spatial_offset=0.00001,
            time_offset=0.00025,
        )

        # Case 2 near in space to result 3, but farther in time.
        self.generate_known_obj_from_result(
            self.known_objs,
            3,  # Base off result 3
            self.obstimes,  # Use all possible obstimes
            "spatial_close_time_far_3",
            spatial_offset=0.0001,
            time_offset=0.3,
        )

        # Case 3: A similar trajectory to result 5, but farther in space with similar timestamps.
        self.generate_known_obj_from_result(
            self.known_objs,
            5,  # Base off result 5
            self.obstimes,  # Use all possible obstimes
            "spatial_far_time_close_5",
            spatial_offset=5,
            time_offset=0.00025,
        )

        # Case 4: A similar trajectory to result 7, but far off spatially and temporally
        self.generate_known_obj_from_result(
            self.known_objs,
            7,  # Base off result 7
            self.obstimes,  # Use all possible obstimes
            "spatial_far_time_far_7",
            spatial_offset=5,
            time_offset=0.3,
        )

        # Case 5: a trajectory matching result 8 but with only a few observations.
        self.generate_known_obj_from_result(
            self.known_objs,
            8,  # Base off result 8
            self.obstimes[::10],  # Samples down to every 5th observation
            "sparse_8",
            spatial_offset=0.0001,
            time_offset=0.00025,
        )

    def test_known_obj_init(
        self,
    ):  # Test a table with no columns specified raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(Table(), self.obstimes, self.filter_params)

        # Test a table with no Name column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"RA": [], "DEC": [], "mjd_mid": []}),
                self.obstimes,
                self.filter_params,
            )

        # Test a table with no RA column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "DEC": [], "mjd_mid": []}),
                self.obstimes,
                self.filter_params,
            )

        # Test a table with no DEC column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "RA": [], "mjd_mid": []}),
                self.obstimes,
                self.filter_params,
            )

        # Test a table with no mjd_mid column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "RA": [], "DEC": []}),
                self.obstimes,
                self.filter_params,
            )

        # Test that we raise errors for when obstimes and filter params are empty
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "RA": [], "DEC": [], "mjd_mid": []}),
                [],
                self.filter_params,
            )
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "RA": [], "DEC": [], "mjd_mid": []}),
                self.obstimes,
                {},
            )

        # Test a table with all columns specified does not raise an error
        correct = KnownObjsMatcher(
            Table({"Name": [], "RA": [], "DEC": [], "mjd_mid": []}),
            self.obstimes,
            self.filter_params,
        )
        self.assertEqual(0, len(correct))

        # Test a table where we override the names for each column
        self.assertEqual(
            0,
            len(
                KnownObjsMatcher(
                    Table({"my_Name": [], "my_RA": [], "my_DEC": [], "my_mjd_mid": []}),
                    self.obstimes,
                    self.filter_params,
                    mjd_col="my_mjd_mid",
                    ra_col="my_RA",
                    dec_col="my_DEC",
                    name_col="my_Name",
                )
            ),
        )

    def generate_known_obj_from_result(
        self,
        known_obj_table,
        res_idx,
        obstimes,
        name,
        spatial_offset=0.0001,
        time_offset=0.00025,
    ):
        """Helper function to generate a known object based on existing result trajectory"""
        trj_skycoords = trajectory_predict_skypos(
            self.res.make_trajectory_list()[res_idx],
            self.wcs,
            obstimes,
        )
        for i in range(len(obstimes)):
            known_obj_table.add_row(
                {
                    "Name": name,
                    "RA": trj_skycoords[i].ra.degree + spatial_offset,
                    "DEC": trj_skycoords[i].dec.degree + spatial_offset,
                    "mjd_mid": obstimes[i] + time_offset,
                }
            )

    def test_apply_known_obj_empty(self):
        # Here we test that the filter across various empty parameters

        # Test that the filter is not applied when no known objects were provided
        empty_objs = KnownObjsMatcher(
            Table({"Name": np.empty(0, dtype=str), "RA": [], "DEC": [], "mjd_mid": []}),
            self.obstimes,
            self.filter_params,
        )
        matches = empty_objs.apply_known_obj_valid_obs_filter(
            self.res,
            wcs=self.wcs,
        )
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

        # Test that the filter is not applied when there were no results.
        matches = empty_objs.apply_known_obj_valid_obs_filter(
            Results(),
            wcs=self.wcs,
        )
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

    def test_apply_known_obj_filtering(self):
        expected_matches = set(["spatial_close_time_close_1", "sparse_8"])

        matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

        # Call the function under test
        matches = matcher.apply_known_obj_valid_obs_filter(
            self.res,
            wcs=self.wcs,
        )

        # Assert the expected result
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        self.assertEqual(9, len(self.res))

        # Check that the close known object we inserted near result 1 is present
        self.assertEqual(len(matches[1]), 1)
        self.assertTrue("spatial_close_time_close_1" in matches[1])

        self.assertEqual(len(matches[8]), 1)
        self.assertTrue("sparse_8" in matches[8])

        # Check that no results other than result 1 have a match
        for i in range(len(self.res)):
            if i != 1 and i != 8:
                self.assertEqual(0, len(matches[i]))

    def test_apply_known_obj_excessive_spatial_filtering(self):
        # Here we only filter for exact spatial matches and should return no results
        self.filter_params["known_obj_sep_thresh"] = 0.0

        matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

        matches = matcher.apply_known_obj_valid_obs_filter(
            self.res,
            wcs=self.wcs,
        )
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

    def test_apply_known_obj_spatial_filtering(self):
        # Here we use a filter that only matches spatially with an unreasonably generous time filter
        self.filter_params["known_obj_sep_time_thresh_s"] = 1000000
        expected_matches = set(["spatial_close_time_close_1", "spatial_close_time_far_3", "sparse_8"])
        matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

        matches = matcher.apply_known_obj_valid_obs_filter(
            self.res,
            wcs=self.wcs,
        )

        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        self.assertEqual(8, len(self.res))

        # Check that the close known object we inserted near result 1 is present
        self.assertEqual(1, len(matches[1]))
        self.assertTrue("spatial_close_time_close_1" in matches[1])
        self.assertEqual(
            np.count_nonzero(self.obs_valid[1]),
            np.count_nonzero(matches[1]["spatial_close_time_close_1"]),
        )

        # Check that the close known object we inserted near result 3 is present
        self.assertEqual(1, len(matches[3]))
        self.assertTrue("spatial_close_time_far_3" in matches[3])
        self.assertEqual(
            np.count_nonzero(self.obs_valid[3]),
            np.count_nonzero(matches[3]["spatial_close_time_far_3"]),
        )

        # Check that the sparse known object we inserted near result 8 is present
        self.assertEqual(1, len(matches[8]))
        self.assertTrue("sparse_8" in matches[8])
        self.assertGreaterEqual(
            len(self.known_objs[self.known_objs["Name"] == "sparse_8"]),
            np.count_nonzero(matches[8]["sparse_8"]),
        )

        # Check that no results other than results 1 and 3 are full matches
        # Since these are based off of random trajectories we can't guarantee there
        # won't some overlapping observations.
        for i in range(len(self.res)):
            if i not in [1, 3]:
                for obj_name in matches[i]:
                    self.assertGreater(
                        np.count_nonzero(self.obs_valid[i]),
                        np.count_nonzero(matches[i][obj_name]),
                    )

    def test_apply_known_obj_temporal_filtering(self):
        # Here we use a filter that only matches temporally with an unreasonably generous spatial filter
        self.filter_params["known_obj_sep_thresh"] = 100000
        expected_matches = set(["spatial_close_time_close_1", "spatial_far_time_close_5", "sparse_8"])
        matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

        matches = matcher.apply_known_obj_valid_obs_filter(
            self.res,
            wcs=self.wcs,
        )

        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)
        self.assertEqual(0, len(self.res))

        # Check that every result matches to  of our known objects
        for i in range(len(matches)):
            self.assertEqual(expected_matches, set(matches[i].keys()))
            # Check that all observations were matched to the known objects
            for obj_name in matches[i]:
                if obj_name == "sparse_8":
                    self.assertGreaterEqual(
                        len(self.known_objs[self.known_objs["Name"] == "sparse_8"]),
                        np.count_nonzero(matches[i]["sparse_8"]),
                    )
                else:
                    self.assertEqual(
                        np.count_nonzero(self.obs_valid[i]),
                        np.count_nonzero(matches[i][obj_name]),
                    )

    def test_apply_known_obj_time_no_filtering(self):
        # Here we use generous temporal and spatial filters to uncover all objects
        self.filter_params["known_obj_sep_thresh"] = 100000
        self.filter_params["known_obj_sep_time_thresh_s"] = 1000000
        expected_matches = set(
            [
                "spatial_close_time_close_1",
                "spatial_close_time_far_3",
                "spatial_far_time_close_5",
                "spatial_far_time_far_7",
                "sparse_8",
            ]
        )
        matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

        matches = matcher.apply_known_obj_valid_obs_filter(
            self.res,
            wcs=self.wcs,
        )
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        # Each result should have matched to every object
        self.assertEqual(0, len(self.res))

        # Check that every result matches to all of expected known objects
        for i in range(len(matches)):
            self.assertEqual(expected_matches, set(matches[i].keys()))
            # Check that all observations were matched to the known objects
            for obj_name in matches[i]:
                self.assertEqual(
                    np.count_nonzero(self.obs_valid[i]),
                    np.count_nonzero(matches[i][obj_name]),
                )

    def test_apply_known_obj_obs_ratio(self):
        min_obs_ratios = [
            0.0,
            1.0,
        ]
        expected_matches = [
            set([]),
            set(["spatial_close_time_close_1", "sparse_8"]),
        ]
        orig_res = self.res.table.copy()
        for min_obs_ratio, expected in zip(min_obs_ratios, expected_matches):
            self.res = Results(data=orig_res.copy())
            self.filter_params["known_obj_match_obs_ratio"] = min_obs_ratio
            matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

            matcher.apply_known_obj_valid_obs_filter(
                self.res,
                wcs=self.wcs,
                update_obs_valid=False,
            )
            # Validate that we did not filter any results
            assert self.filter_params["filter_type"] in self.res.table.columns
            self.assertEqual(10, len(self.res))

            # Generate the recovered ratio column
            matcher.apply_known_obj_match_obs_ratio(self.res)
            match_col = "recovered_test_matches_obs_ratio"
            assert match_col in self.res.table.columns

            # Verify that we recovered the expected matches
            recovered_matches = set()
            for i in range(len(self.res)):
                recovered_matches.update(self.res[i][match_col])
            self.assertEqual(expected, recovered_matches)

            # Verify that we filter out our expected results
            matcher.filter_known_obj(self.res, match_col, match_col)
            self.assertEqual(10 - len(expected), len(self.res))

    def test_apply_known_obj_min_obs(self):
        min_obs_settings = [
            100,
            1,
            5,
        ]
        expected_matches = [
            set([]),
            set(["spatial_close_time_close_1", "sparse_8"]),
            set(["spatial_close_time_close_1"]),
        ]
        orig_res = self.res.table.copy()
        for min_obs, expected in zip(min_obs_settings, expected_matches):
            self.res = Results(data=orig_res.copy())
            self.filter_params["known_obj_match_min_obs"] = min_obs
            matcher = KnownObjsMatcher(self.known_objs, self.obstimes, self.filter_params)

            matcher.apply_known_obj_valid_obs_filter(
                self.res,
                wcs=self.wcs,
                update_obs_valid=False,
            )
            # Validate that we did not filter any results
            assert self.filter_params["filter_type"] in self.res.table.columns
            self.assertEqual(10, len(self.res))

            # Generate the recovered object column for a minimum number of observations
            matcher.apply_known_obj_match_min_obs(self.res)
            match_col = "recovered_test_matches_min_obs"
            assert match_col in self.res.table.columns

            # Verify that we recovered the expected matches
            recovered_matches = set()
            for i in range(len(self.res)):
                recovered_matches.update(self.res[i][match_col])
            if expected != recovered_matches:
                raise ValueError(f"Expected {expected} but got {recovered_matches} for min_obs={min_obs}")
            self.assertEqual(expected, recovered_matches)

            # Verify that we filter out our expected results
            matcher.filter_known_obj(self.res, match_col, match_col)
            self.assertEqual(10 - len(expected), len(self.res))
