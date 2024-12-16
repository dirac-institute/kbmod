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


class TestKnownObjMatcher(unittest.TestCase):
    def setUp(self):
        # Seed for reproducibility of random generated trajectories
        self.seed = 500
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Set up some default parameters for our matcher
        self.matcher_name = "test_matches"
        self.sep_thresh = 1.0
        self.time_thresh_s = 600.0

        # Create a fake dataset with 15 x 10 images and 25 obstimes.
        num_images = 25
        self.obstimes = np.array(create_fake_times(num_images))
        ds = FakeDataSet(15, 10, self.obstimes, use_seed=True)
        self.wcs = make_fake_wcs(10.0, 15.0, 15, 10)
        ds.set_wcs(self.wcs)

        # Randomly generate a Trajectory for each of our 10 results
        num_results = 10
        for i in range(num_results):
            ds.insert_random_object(self.seed)
        self.res = Results.from_trajectories(ds.trajectories, track_filtered=True)
        self.assertEqual(len(ds.trajectories), num_results)

        # Generate which observations are valid observations for each result
        self.obs_valid = np.full((num_results, num_images), True)
        for i in range(num_results):
            # For each result include a random set of 5 invalid observations
            invalid_obs = np.random.choice(num_images, 5, replace=False)
            self.obs_valid[i][invalid_obs] = False
        self.res.update_obs_valid(self.obs_valid)
        assert set(self.res.table.columns) == set(
            ["x", "y", "vx", "vy", "likelihood", "flux", "obs_count", "obs_valid"]
        )

        # Use the results' trajectories to generate a set of known objects that intersect our generated results in various
        # ways.
        self.known_objs = Table({"Name": np.empty(0, dtype=str), "RA": [], "DEC": [], "mjd_mid": []})

        # Have the temporal offset for near and far objects be just below and above our time threshold
        time_offset_mjd_close = (self.time_thresh_s - 1) / (24.0 * 3600)
        time_offset_mjd_far = (self.time_thresh_s + 1) / (24.0 * 3600)

        # Case 1: Near in space and near in time just within the range of our filters to result 1
        self.generate_known_obj_from_result(
            self.known_objs,
            1,  # Base off result 1
            self.obstimes,  # Use all possible obstimes
            "spatial_close_time_close_1",
            spatial_offset=0.00001,
            time_offset=time_offset_mjd_close,
        )

        # Case 2 near in space to result 3, but farther in time.
        self.generate_known_obj_from_result(
            self.known_objs,
            3,  # Base off result 3
            self.obstimes,  # Use all possible obstimes
            "spatial_close_time_far_3",
            spatial_offset=0.0001,
            time_offset=time_offset_mjd_far,
        )

        # Case 3: A similar trajectory to result 5, but farther in space with similar timestamps.
        self.generate_known_obj_from_result(
            self.known_objs,
            5,  # Base off result 5
            self.obstimes,  # Use all possible obstimes
            "spatial_far_time_close_5",
            spatial_offset=5,
            time_offset=time_offset_mjd_close,
        )

        # Case 4: A similar trajectory to result 7, but far off spatially and temporally
        self.generate_known_obj_from_result(
            self.known_objs,
            7,  # Base off result 7
            self.obstimes,  # Use all possible obstimes
            "spatial_far_time_far_7",
            spatial_offset=5,
            time_offset=time_offset_mjd_far,
        )

        # Case 5: a trajectory matching result 8 but with only a few observations.
        self.generate_known_obj_from_result(
            self.known_objs,
            8,  # Base off result 8
            self.obstimes[::10],  # Samples down to every 10th observation
            "sparse_8",
            spatial_offset=0.0001,
            time_offset=time_offset_mjd_close,
        )

    def test_known_objs_matcher_init(
        self,
    ):  # Test that a table with no columns specified raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table(),
                self.obstimes,
                self.matcher_name,
                self.sep_thresh,
                self.time_thresh_s,
            )

        # Test that a table with no Name column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"RA": [], "DEC": [], "mjd_mid": []}),
                self.obstimes,
                self.matcher_name,
                self.sep_thresh,
                self.time_thresh_s,
            )

        # Test that a table with no RA column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "DEC": [], "mjd_mid": []}),
                self.obstimes,
                self.matcher_name,
                self.sep_thresh,
                self.time_thresh_s,
            )

        # Test that a table with no DEC column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "RA": [], "mjd_mid": []}),
                self.obstimes,
                self.matcher_name,
                self.sep_thresh,
                self.time_thresh_s,
            )

        # Test that a table with no mjd_mid column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjsMatcher(
                Table({"Name": [], "RA": [], "DEC": []}),
                self.obstimes,
                self.matcher_name,
                self.sep_thresh,
                self.time_thresh_s,
            )

        # Test that a table with all columns specified does not raise an error
        correct = KnownObjsMatcher(
            Table({"Name": [], "RA": [], "DEC": [], "mjd_mid": []}),
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )
        self.assertEqual(0, len(correct))

        # Test a table where we override the names for each column
        self.assertEqual(
            0,
            len(
                KnownObjsMatcher(
                    Table({"my_Name": [], "my_RA": [], "my_DEC": [], "my_mjd_mid": []}),
                    self.obstimes,
                    self.matcher_name,
                    self.sep_thresh,
                    self.time_thresh_s,
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
        """Helper function to generate a known object based on an existing result trajectory"""
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

    def test_known_objs_match_empty(self):
        # Here we test the filter across various empty parameters

        # Test that the filter is not applied when no known objects were provided
        empty_objs = KnownObjsMatcher(
            Table({"Name": np.empty(0, dtype=str), "RA": [], "DEC": [], "mjd_mid": []}),
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )
        self.res = empty_objs.match(
            self.res,
            self.wcs,
        )
        # Though there were no known objects, check that the results table still has rows
        self.assertEqual(10, len(self.res))
        # We should still apply the matching column to the results table even if empty
        matches = self.res[empty_objs.matcher_name]
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))

        # Test that we can apply the filter even when there are known results
        self.res = empty_objs.mark_matched_obs_invalid(self.res, drop_empty_rows=True)
        self.assertEqual(10, len(self.res))

        # Test that the filter is not applied when there were no results.
        empty_res = Results()
        empty_res = empty_objs.match(
            empty_res,
            self.wcs,
        )
        matches = empty_res[empty_objs.matcher_name]
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))

        empty_res = empty_objs.mark_matched_obs_invalid(empty_res, drop_empty_rows=True)
        self.assertEqual(0, len(empty_res))

    def test_match(self):
        # We expect to find only the objects close in time and space to our results,
        # including one object matching closely to a result across all observations
        # and also a sparsely represented object with only a few observations.
        expected_matches = set(["spatial_close_time_close_1", "sparse_8"])
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )

        # Generate matches for the results according to the known objects
        self.res = matcher.match(
            self.res,
            self.wcs,
        )
        matches = self.res[self.matcher_name]
        # Assert the expected result
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        # Check that the close known object we inserted near result 1 is dropped
        # But the sparsely observed known object will not get filtered out.
        self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=True)
        self.assertEqual(9, len(self.res))

        # Check that the close known object we inserted near result 1 is present
        self.assertEqual(len(matches[1]), 1)
        self.assertTrue("spatial_close_time_close_1" in matches[1])

        self.assertEqual(len(matches[8]), 1)
        self.assertTrue("sparse_8" in matches[8])

        # Check that no results other than results 1 and 8 have a match
        for i in range(len(self.res)):
            if i != 1 and i != 8:
                self.assertEqual(0, len(matches[i]))

    def test_match_excessive_spatial_filtering(self):
        # Here we only filter for exact spatial matches and should return no results
        self.sep_thresh = 0.0
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )

        self.res = matcher.match(
            self.res,
            self.wcs,
        )
        matches = self.res[matcher.matcher_name]
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))

        self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=True)
        self.assertEqual(10, len(self.res))

    def test_match_spatial_filtering(self):
        # Here we use a filter that only matches spatially with an unreasonably generous time filter
        self.time_thresh_s += 2
        # Our expected matches now include all objects that are close in space to our results regardless
        # of the time offset we generated.
        expected_matches = set(["spatial_close_time_close_1", "spatial_close_time_far_3", "sparse_8"])
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )

        # Performing matching
        self.res = matcher.match(
            self.res,
            self.wcs,
        )
        matches = self.res[matcher.matcher_name]

        # Confirm that the expected matches are present
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        # Check that the close known objects we inserted are removed by valid obs filtering
        # while the sparse known object does not fully filter out that result.
        self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=True)
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

    def test_match_temporal_filtering(self):
        # Here we use a filter that only matches temporally with an unreasonably generous spatial filter
        self.sep_thresh = 100000
        expected_matches = set(["spatial_close_time_close_1", "spatial_far_time_close_5", "sparse_8"])
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )

        # Generate matches
        self.res = matcher.match(
            self.res,
            self.wcs,
        )
        matches = self.res[matcher.matcher_name]

        # Confirm that the expected matches are present
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        # Because we have objects that match to each observation temporally,
        # a generous spatial filter will filter out all valid observations.
        self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=True)
        self.assertEqual(0, len(self.res))

        for i in range(len(matches)):
            self.assertEqual(expected_matches, set(matches[i].keys()))
            # Check that all observations were matched to the known objects
            for obj_name in matches[i]:
                if obj_name == "sparse_8":
                    # The sparse object only has a few observations to match
                    self.assertGreaterEqual(
                        len(self.known_objs[self.known_objs["Name"] == "sparse_8"]),
                        np.count_nonzero(matches[i]["sparse_8"]),
                    )
                else:
                    # The other objects have a full set of observations to match
                    self.assertEqual(
                        np.count_nonzero(self.obs_valid[i]),
                        np.count_nonzero(matches[i][obj_name]),
                    )

    def test_match_all(self):
        # Here we use generous temporal and spatial filters to recover all objects
        self.sep_thresh = 100000
        self.time_thresh_s = 1000000
        expected_matches = set(
            [
                "spatial_close_time_close_1",
                "spatial_close_time_far_3",
                "spatial_far_time_close_5",
                "spatial_far_time_far_7",
                "sparse_8",
            ]
        )
        # Perform the matching
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
            self.sep_thresh,
            self.time_thresh_s,
        )
        self.res = matcher.match(
            self.res,
            self.wcs,
        )

        # Here we expect to recover all of our known objects.
        matches = self.res[matcher.matcher_name]
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        # Each result should have matched to every object
        self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=True)
        self.assertEqual(0, len(self.res))

        # Check that every result matches to all of expected known objects
        for i in range(len(matches)):
            self.assertEqual(expected_matches, set(matches[i].keys()))
            # Check that all observations were matched to the known objects since
            # ven the most sparse object should match to every observation with
            # our time filter.
            for obj_name in matches[i]:
                self.assertEqual(
                    np.count_nonzero(self.obs_valid[i]),
                    np.count_nonzero(matches[i][obj_name]),
                )

    def test_match_obs_ratio_invalid(self):
        # Here we test that we raise an error for observation ratios outside of the valid range
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
        )
        self.res = matcher.match(self.res, self.wcs)

        # Test some inavlid ratios outside of the range [0, 1]
        with self.assertRaises(ValueError):
            matcher.match_on_obs_ratio(self.res, 1.1)
        with self.assertRaises(ValueError):
            matcher.match_on_obs_ratio(self.res, -0.1)

    def test_match_obs_ratio(self):
        # Here we test considering a known object recovered based on the ratio of observations
        # in the catalog that were temporally within
        min_obs_ratios = [
            0.0,
            1.0,
        ]
        # The expected matching objects for each min_obs_ratio parameter chosen.
        expected_matches = [
            set([]),
            set(["spatial_close_time_close_1", "sparse_8"]),
        ]
        orig_res = self.res.table.copy()
        for obs_ratio, expected in zip(min_obs_ratios, expected_matches):
            self.res = Results(data=orig_res.copy())
            matcher = KnownObjsMatcher(
                self.known_objs,
                self.obstimes,
                matcher_name=self.matcher_name,
                sep_thresh=self.sep_thresh,
                time_thresh_s=self.time_thresh_s,
            )

            # Perform the intial matching
            self.res = matcher.match(
                self.res,
                self.wcs,
            )

            # Validate that we did not filter any results by obstimes
            assert self.matcher_name in self.res.table.columns
            self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=False)
            self.assertEqual(10, len(self.res))

            # Generate the column of which objects were "recovered"
            matcher.match_on_obs_ratio(self.res, obs_ratio)
            match_col = f"recovered_test_matches_obs_ratio_{obs_ratio}"
            assert match_col in self.res.table.columns
            assert match_col == matcher.match_obs_ratio_col(obs_ratio)

            # Verify that we recovered the expected matches
            recovered, missed = matcher.get_recovered_objects(
                self.res, matcher.match_obs_ratio_col(obs_ratio)
            )
            self.assertEqual(expected, recovered)
            # The missed object are all other known objects in our catalog - the expected objects
            expected_missed = set(self.known_objs["Name"]) - expected
            self.assertEqual(expected_missed, missed)

            # Verify that we filter out our expected results
            matcher.filter_matches(self.res, match_col)
            self.assertEqual(10 - len(expected), len(self.res))

    def test_match_min_obs(self):
        # Here we test considering a known object recovered based on the ratio of observations
        # in the catalog that were temporally within
        min_obs_settings = [
            100,  # No objects should be recovered since our catalog objects have fewer observations
            1,
            5,  # The sparse object will not have enough observations to be recovered.
        ]
        expected_matches = [
            set([]),
            set(["spatial_close_time_close_1", "sparse_8"]),
            set(["spatial_close_time_close_1"]),
        ]
        orig_res = self.res.table.copy()
        for min_obs, expected in zip(min_obs_settings, expected_matches):
            self.res = Results(data=orig_res.copy())
            matcher = KnownObjsMatcher(
                self.known_objs,
                self.obstimes,
                matcher_name=self.matcher_name,
                sep_thresh=self.sep_thresh,
                time_thresh_s=self.time_thresh_s,
            )
            # Perform the initial matching
            matcher.match(
                self.res,
                self.wcs,
            )
            # Validate that we did not filter any results
            assert self.matcher_name in self.res.table.columns
            self.res = matcher.mark_matched_obs_invalid(self.res, drop_empty_rows=False)
            self.assertEqual(10, len(self.res))

            # Generate the recovered object column for a minimum number of observations
            matcher.match_on_min_obs(self.res, min_obs)
            match_col = f"recovered_test_matches_min_obs_{min_obs}"
            assert match_col in self.res.table.columns
            assert match_col == matcher.match_min_obs_col(min_obs)

            # Verify that we recovered the expected matches
            recovered, missed = matcher.get_recovered_objects(self.res, matcher.match_min_obs_col(min_obs))
            self.assertEqual(expected, recovered)
            # The missed object are all other known objects in our catalog - the expected objects
            expected_missed = set(self.known_objs["Name"]) - expected
            self.assertEqual(expected_missed, missed)

            # Verify that we filter out our expected results
            matcher.filter_matches(self.res, match_col)
            self.assertEqual(10 - len(expected), len(self.res))

    def test_empty_filter_matches(self):
        # Test that we can filter matches with an empty Results table
        empty_res = Results()
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
        )

        # No matcher_name column in the data.
        with self.assertRaises(ValueError):
            _ = matcher.match_on_obs_ratio(empty_res, 0.5)

        # Do the match to add the columns.
        matcher.match(empty_res, self.wcs)
        empty_res = matcher.match_on_obs_ratio(empty_res, 0.5)

        # Test an invalid matching column
        with self.assertRaises(ValueError):
            matcher.filter_matches(empty_res, "empty")

        empty_res = matcher.filter_matches(empty_res, matcher.match_obs_ratio_col(0.5))
        self.assertEqual(0, len(empty_res))

    def test_empty_get_recovered_objects(self):
        # Test that we can get recovered objects with an empty Results table
        empty_res = Results()
        matcher = KnownObjsMatcher(
            self.known_objs,
            self.obstimes,
            self.matcher_name,
        )

        # No matcher_name column in the data.
        with self.assertRaises(ValueError):
            _ = matcher.match_on_min_obs(empty_res, 5)

        # Do the match to add the columns.
        matcher.match(empty_res, self.wcs)
        empty_res = matcher.match_on_min_obs(empty_res, 5)

        # Test an invalid matching column
        with self.assertRaises(ValueError):
            matcher.get_recovered_objects(empty_res, "empty")

        recovered, missed = matcher.get_recovered_objects(empty_res, matcher.match_min_obs_col(5))
        self.assertEqual(0, len(recovered))
        self.assertEqual(0, len(missed))


if __name__ == "__main__":
    unittest.main()
