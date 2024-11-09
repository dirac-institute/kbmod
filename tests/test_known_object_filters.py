import unittest
from unittest.mock import MagicMock
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from kbmod.results import Results
from kbmod.work_unit import WorkUnit
from kbmod.filters.known_object_filters import apply_known_obj_filters, KnownObjs
import unittest
from unittest.mock import MagicMock

import random

from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod.trajectory_utils import trajectory_predict_skypos

from kbmod.fake_data.fake_data_creator import *
from kbmod.search import *
from kbmod.wcs_utils import make_fake_wcs, wcs_fits_equal

class TestKnownObjFilters(unittest.TestCase):
    def setUp(self):
        self.seed = 500 # Seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.filter_params = {
            "filter_type": "known_obj_matches",
            "known_objs_filepath": "/path/to/known_objs.csv",
            "known_obj_thresh": 0.5,
            "known_obj_sep_thresh": 1.0,
            "known_obj_sep_time_thresh_s": 600,
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
        self.res = Results.from_trajectories(ds.trajectories)
        self.assertEqual(len(ds.trajectories), num_results)

        # Generate which observations are valid observations for each result
        self.obs_valid = np.full((num_results, num_images), True)
        for i in range(num_results):
            invalid_obs = np.random.choice(num_images, 5, replace=False)
            self.obs_valid[i][invalid_obs] = False
        self.res.update_obs_valid(self.obs_valid)
        assert set(self.res.table.columns) == set(['x','y','vx','vy','likelihood','flux','obs_count', 'obs_valid'])

        # Use the results' trajectories to generate a set of known objects that we can use to test the filter
        # 1) a known object that is slightly different in time and space then in the trajcectory
        # 2) a known object that is slightly different in time but not space
        # 3) a known object that is slightly different in space but not time
        # 4) a known object with a very different trajectory but interects both the start and endpoints
        # Now we want to create a data set of known objects some
        # of which will intersect or previous results, 
        known_obj_table = Table({"Name": np.empty(0, dtype=str), "RA": [], "DEC": [], "mjd_mid": []})

        # Case 1: Near in space (<1") and near in time (>1 s) and near in time to result 1
        self.generate_known_obj_from_result(
            known_obj_table,
            1, # Base off result 1
            self.obstimes, # Use all possible obstimes 
            "spatial_close_time_close_1",
            spatial_offset=0.00001,
            time_offset=0.00025)

        # Case 2 near in space to result 3, but farther in time.
        self.generate_known_obj_from_result(
            known_obj_table,
            3, # Base off result 3
            self.obstimes, # Use all possible obstimes 
            "spatial_close_time_far_3",
            spatial_offset=0.0001,
            time_offset=0.3)

        # Case 3: A similar trajectory to result 5, but farther in space with similar timestamps.
        self.generate_known_obj_from_result(
            known_obj_table,
            5, # Base off result 5
            self.obstimes, # Use all possible obstimes
            "spatial_far_time_close_5",
            spatial_offset=5,
            time_offset=0.00025)

        # Case 5: A similar trajectory to result 7, but far off spatially and temporally
        self.generate_known_obj_from_result(
            known_obj_table,
            7, # Base off result 7
            self.obstimes, # Use all possible obstimes
            "spatial_far_time_far_7",
            spatial_offset=5,
            time_offset=0.3)


        # Check for a trajectory matching result 8 but with only a few observations.
        self.generate_known_obj_from_result(
            known_obj_table,
            8, # Base off result 8
            self.obstimes[::10], # Samples down to every 5th observation
            "sparse_8",
            spatial_offset=0.0001,
            time_offset=0.00025)
        """
        curr_trj_skycoords = trajectory_predict_skypos(
            self.res.make_trajectory_list()[8],
            self.wcs,
            self.obstimes, # Note that we use all obstimes and not just the valid ones for this result.
        )
        for i in range(len(self.obstimes)):
            known_obj_table.add_row({
                "Name": "sparse_8",
                "RA": curr_trj_skycoords[i].ra.degree,
                "DEC": curr_trj_skycoords[i].dec.degree, 
                "mjd_mid": self.obstimes[i],
            })
        """

        # Check that intersect 1 and intersect 2 both match to result 7, but only to one observation
        # each


        # Check for an object that is an exact spatial match but omitted is outside the larger time
        # window we apply during the filtering steps.
        # Check for the max time step separation



        self.known_objs = KnownObjs(known_obj_table)
    def test_known_obj_init(self):
        # Test a table with no columns specified raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjs(Table())

        # Test a table with no Name column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjs(Table({"RA": [], "DEC": [], "mjd_mid": []}))

        # Test a table with no RA column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjs(Table({"Name": [], "DEC": [], "mjd_mid": []}))
        
        # Test a table with no DEC column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjs(Table({"Name": [], "RA": [], "mjd_mid": []}))

        # Test a table with no mjd_mid column raises a ValueError
        with self.assertRaises(ValueError):
            KnownObjs(Table({"Name": [], "RA": [], "DEC": []}))

        # Test a table with all columns specified does not raise an error
        self.assertEqual(0, len(KnownObjs(Table({"Name": [], "RA": [], "DEC": [], "mjd_mid": []}))))

        # Test a table where we override the names for each column
        self.assertEqual(0, len(KnownObjs(
            Table({"my_Name": [], "my_RA": [], "my_DEC": [], "my_mjd_mid": []}),
            mjd_col="my_mjd_mid",
            ra_col="my_RA",
            dec_col="my_DEC", 
            name_col="my_Name",
        )))

    def generate_known_obj_from_result(
            self,
            known_obj_table,
            res_idx,
            obstimes,
            name,
            spatial_offset=0.0001,
            time_offset=0.00025,
        ):
        """ Helper function to generate a known object based on existing result trajectory 
        Parameters
        ----------

        """
        trj_skycoords = trajectory_predict_skypos(
            self.res.make_trajectory_list()[res_idx],
            self.wcs,
            obstimes, 
        )
        for i in range(len(obstimes)):
            known_obj_table.add_row({
                "Name": name,
                "RA": trj_skycoords[i].ra.degree + spatial_offset,
                "DEC": trj_skycoords[i].dec.degree + spatial_offset, 
                "mjd_mid": obstimes[i] + time_offset,
            })

    def test_known_obj_filter_rows_by_time(self):
        assert True

    def test_known_obj_filter_wrapper(self):
        assert True

    def test_apply_known_obj_empty(self):
        # Here we test that the filter across various empty parameters

        # Test that the filter is not applied when no known objects were provided
        empty_objs = KnownObjs(
            Table({"Name": np.empty(0, dtype=str), "RA": [], "DEC": [], "mjd_mid": []}))
        matches = apply_known_obj_filters(self.res, empty_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

        # Test that the filter is not applied when there were no results.
        matches = apply_known_obj_filters(Results(), self.known_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

        # Test that the filter is not applied when there were no obstimes
        matches = apply_known_obj_filters(self.res, self.known_objs, obstimes=[], wcs=self.wcs, filter_params=self.filter_params)
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

    def test_apply_known_obj_filtering(self):
        expected_matches = set(["spatial_close_time_close_1", "sparse_8"])

        # Call the function under test
        matches = apply_known_obj_filters(self.res, self.known_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
        
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
        matches = apply_known_obj_filters(
            self.res, self.known_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
        self.assertEqual(0, sum([len(m.keys()) for m in matches]))
        self.assertEqual(10, len(self.res))

    def test_apply_known_obj_spatial_filtering(self):
        # Here we use a filter that only matches spatially with an unreasonably generous time filter
        self.filter_params["known_obj_sep_time_thresh_s"] = 1000000
        expected_matches = set([
            "spatial_close_time_close_1",
            "spatial_close_time_far_3",
            "sparse_8"])

        matches = apply_known_obj_filters(
            self.res, self.known_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
    
        obs_matches = set()
        for m in matches:
            obs_matches.update(m.keys())
        self.assertEqual(expected_matches, obs_matches)

        self.assertEqual(8, len(self.res))

        # Check that the close known object we inserted near result 1 is present
        self.assertEqual(1, len(matches[1]))
        self.assertTrue("spatial_close_time_close_1" in matches[1])
        self.assertEqual(np.count_nonzero(self.obs_valid[1]),
                            np.count_nonzero(matches[1]["spatial_close_time_close_1"]))

        # Check that the close known object we inserted near result 3 is present
        self.assertEqual(1, len(matches[3]))
        self.assertTrue("spatial_close_time_far_3" in matches[3])
        self.assertEqual(np.count_nonzero(self.obs_valid[3]),
                        np.count_nonzero(matches[3]["spatial_close_time_far_3"]))
        
        # Check that the sparse known object we inserted near result 8 is present
        self.assertEqual(1, len(matches[8]))
        self.assertTrue("sparse_8" in matches[8])
        self.assertGreaterEqual(len(self.known_objs.data[self.known_objs.data["Name"] == "sparse_8"]),
                        np.count_nonzero(matches[8]["sparse_8"]))

        # Check that no results other than results 1 and 3 are full matches
        # Since these are based off of random trajectories we can't guarantee there
        # won't some overlapping observations.
        for i in range(len(self.res)):
            if i not in [1, 3]:
                for obj_name in matches[i]:
                    self.assertGreater(
                            np.count_nonzero(self.obs_valid[i]),
                            np.count_nonzero(matches[i][obj_name]))

    def test_apply_known_obj_temporal_filtering(self):
        # Here we use a filter that only matches temporally with an unreasonably generous spatial filter
        self.filter_params["known_obj_sep_thresh"] = 100000
        expected_matches = set([
            "spatial_close_time_close_1",
            "spatial_far_time_close_5",
            "sparse_8"])
        
        matches = apply_known_obj_filters(
            self.res, self.known_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
        
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
                        len(self.known_objs.data[self.known_objs.data["Name"] == "sparse_8"]),
                        np.count_nonzero(matches[i]["sparse_8"]))
                else:
                    self.assertEqual(
                            np.count_nonzero(self.obs_valid[i]),
                            np.count_nonzero(matches[i][obj_name])
                        )

    def test_apply_known_obj_time_no_filtering(self):
        # Here we use generous temporal and spatial filters to uncover all objects
        self.filter_params["known_obj_sep_thresh"] = 100000
        self.filter_params["known_obj_sep_time_thresh_s"] = 1000000
        expected_matches = set([
            "spatial_close_time_close_1",
            "spatial_close_time_far_3",
            "spatial_far_time_close_5",
            "spatial_far_time_far_7",
            "sparse_8"])
        
        matches = apply_known_obj_filters(
            self.res, self.known_objs, obstimes=self.obstimes, wcs=self.wcs, filter_params=self.filter_params)
        
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
                        np.count_nonzero(matches[i][obj_name])
                    )

    def test_apply_known_obj_matching(self):
        # Here we apply the filter successively without removing matching observations,
        # then successively pare down observations more aggressively.
        assert True
