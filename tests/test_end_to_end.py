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

# from utils.utils_for_tests import get_absolute_demo_data_path


####
import unittest
import itertools
import random

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from astropy.time import Time
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from kbmod import ImageCollection
from kbmod.run_search import SearchRunner
from kbmod.configuration import SearchConfiguration
from kbmod.reprojection import reproject_work_unit
import kbmod.mocking as kbmock


class TestEmptySearch(unittest.TestCase):
    def setUp(self):
        self.factory = kbmock.EmptyFits()

    def test_empty(self):
        """Test no detections are found on empty images."""
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
        """Test no detections are found on images containing static objects."""
        src_cat = kbmock.SourceCatalog.from_defaults(seed=100)
        factory = kbmock.SimpleFits(src_cat=src_cat)
        hduls = factory.mock(10)

        ic = ImageCollection.fromTargets(hduls, force="TestDataStd")
        wu = ic.toWorkUnit(search_config=SearchConfiguration())
        results = SearchRunner().run_search_from_work_unit(wu)
        self.assertTrue(len(results) == 0)


class TestRandomLinearSearch(unittest.TestCase):
    def setUp(self):
        # Set up shared search values
        self.n_imgs = 5
        self.repeat_n_times = 10
        self.shape = (200, 200)
        self.start_pos = (85, 115)
        self.vxs = [-20, 20]
        self.vys = [-20, 20]

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
                    "vx_steps": 40,
                    "vy_steps": 40,
                },
                "num_obs": self.n_imgs,
                "do_mask": False,
                "do_clustering": True,
                "do_stamp_filter": False,
            }
        )

    def xmatch_best(self, obj, results, match_cols={"x_mean": "x", "y_mean": "y", "vx": "vx", "vy": "vy"}):
        """Finds the result that minimizes the L2 distance to the target object.

        Parameters
        ----------
        obj : `astropy.table.Row`
            Row, or a table with single entry, containing the target object.
        results : `astropy.table.Table`
            Table of objects from which the closest matching one will be returned.
        match_cols : `dict`, optional
            Dictionary of column names on which to perform the matching. Keys
            of the dictionary are columns from ``obj`` and values of the dict
            are columns from ``results``.

        Returns
        -------
        result : `astropy.table.Row`
            Best matching result
        distances: `np.array`
            Array of calculated L2 distances of ``obj`` to all given results.
        """
        objk, resk = [], []
        for k, v in match_cols.items():
            if k in obj.columns and v in results.table.columns:
                objk.append(k)
                resk.append(v)
        tgt = np.fromiter(obj[tuple(objk)].values(), dtype=float, count=len(objk))
        res = structured_to_unstructured(results[tuple(resk)].as_array(), dtype=float)
        diff = np.linalg.norm(tgt - res, axis=1)
        if len(results) == 1:
            return results[0], diff
        return results[diff == diff.min()][0], diff

    def assertResultValuesWithinSpec(
        self, expected, result, spec, match_cols={"x_mean": "x", "y_mean": "y", "vx": "vx", "vy": "vy"}
    ):
        """Asserts expected object matches the given result object within
        specification.

        Parameters
        ----------
        expected : `astropy.table.Row`
            Row, or table with single entry, containing the target object.
        result : `astropy.table.Row`
            Row, or table with single entry, containing the found object.
        spec : `float`
            Specification of maximum deviation of the expected values from the
            found resulting values. For example, a spec of 3 means results can
            be 3 or less pixels away from the expected position.
        match_cols : `dict`, optional
            Dictionary of column names on which to perform the matching. Keys
            of the dictionary are columns from ``obj`` and values of the dict
            are columns from ``results``.

        Raises
        -------
        AssertionError - if comparison fails.
        """
        for ekey, rkey in match_cols.items():
            info = (
                f"\n Expected: \n {expected[tuple(match_cols.keys())]} \n"
                f"Retrieved : \n {result[tuple(match_cols.values())]}"
            )
            self.assertLessEqual(abs(expected[ekey] - result[rkey]), spec, info)

    def run_single_search(self, data, expected, spec=5):
        """Runs a KBMOD search on given data and tests the results lie within
        specification from the expected.

        Parameters
        ----------
        data : `list[str]` or `list[astropy.io.fits.HDUList]`
            List of targets processable by the TestDataStandardizer.
        expected : `kbmod.mocking.ObjectCatalog`
            Object catalog expected to be retrieved from the run.
        spec : `float`
            Specification of maximum deviation of the expected values from the
            found resulting values. For example, a spec of 3 means results can
            be 3 or less pixels away from the expected position.
        """
        ic = ImageCollection.fromTargets(data, force="TestDataStd")
        wu = ic.toWorkUnit(search_config=self.config)
        results = SearchRunner().run_search_from_work_unit(wu)

        # Run tests
        self.assertGreaterEqual(len(results), 1)
        for obj in expected.table:
            res, dist = (results[0], None) if len(results) == 1 else self.xmatch_best(obj, results)
            self.assertResultValuesWithinSpec(obj, res, spec)

    def test_exact_motion(self):
        """Test exact searches are recovered in all 8 cardinal directions."""
        search_vs = list(itertools.product([-20, 0, 20], repeat=2))
        search_vs.remove((0, 0))
        for vx, vy in search_vs:
            with self.subTest(f"Cardinal direction: {(vx, vy)}"):
                self.config._params["generator_config"] = {"name": "SingleVelocitySearch", "vx": vx, "vy": vy}
                obj_cat = kbmock.ObjectCatalog.from_defaults(self.param_ranges, n=1)
                obj_cat.table["vx"] = vx
                obj_cat.table["vy"] = vy
                factory = kbmock.SimpleFits(shape=self.shape, step_t=1, obj_cat=obj_cat)
                hduls = factory.mock(n=self.n_imgs)
                self.run_single_search(hduls, obj_cat, 1)

    def test_random_motion(self):
        """Repeat searches for randomly inserted objects."""
        # Mock the data and repeat tests. The random catalog
        # creation guarantees a diverse set of changing test values
        for i in range(self.repeat_n_times):
            with self.subTest(f"Iteration {i}"):
                obj_cat = kbmock.ObjectCatalog.from_defaults(self.param_ranges, n=1)
                factory = kbmock.SimpleFits(shape=self.shape, step_t=1, obj_cat=obj_cat)
                hduls = factory.mock(n=self.n_imgs)
                self.run_single_search(hduls, obj_cat)

    def test_resampled_search(self):
        """Search for objects in a set of resampled images; randomly dithered pointings and orientations."""
        # 0. Setup
        self.shape = (500, 500)
        self.start_pos = (10, 10)  # (ra, dec) in deg
        n_obj = 1
        pixscale = 0.2
        timestamps = Time(np.arange(58915, 58915 + self.n_imgs, 1), format="mjd")
        vx = 0.001  # degrees / day (given the timestamps)
        vy = 0.001

        # 1. Mock data
        #    - mock catalogs, set expected positions by hand
        #    - mock WCSs so that they dither around (10, 10)
        #    - instantiate the required mockers and mock
        cats = []
        for i, t in enumerate(timestamps):
            cats.append(
                Table(
                    {
                        "amplitude": [100],
                        "obstime": [t],
                        "ra_mean": [self.start_pos[0] + vx * i],
                        "dec_mean": [self.start_pos[1] + vy * i],
                        "stddev": [2.0],
                    }
                )
            )
        catalog = vstack(cats)
        obj_cat = kbmock.ObjectCatalog.from_table(catalog, kind="world", mode="folding")

        wcs_factory = kbmock.WCSFactory(
            pointing=self.start_pos,
            rotation=0,
            pixscale=pixscale,
            dither_pos=True,
            dither_rot=True,
            dither_amplitudes=(0.001, 0.001, 10),
        )

        prim_hdr_factory = kbmock.HeaderFactory.from_primary_template(
            mutables=["DATE-OBS"],
            callbacks=[kbmock.ObstimeIterator(timestamps)],
        )

        factory = kbmock.SimpleFits(shape=self.shape, obj_cat=obj_cat, wcs_factory=wcs_factory)
        factory.prim_hdr = prim_hdr_factory
        hduls = factory.mock(n=self.n_imgs)

        # 2. Run search
        #    - make an IC
        #    - determine WCS footprint to reproject to
        #    - determine the pixel-based velocity to search for
        #    - reproject
        #    - run search
        ic = ImageCollection.fromTargets(hduls, force="TestDataStd")

        from reproject.mosaicking import find_optimal_celestial_wcs

        opt_wcs, self.shape = find_optimal_celestial_wcs(list(ic.wcs))
        opt_wcs.array_shape = self.shape

        meanvx = -vx * 3600 / pixscale
        meanvy = vy * 3600 / pixscale

        # The velocity grid needs to be searched very densely for the realistic
        # case (compared to the fact the velocity spread is not that large), and
        # we'll still end up ~10 pixels away from the truth.
        search_config = SearchConfiguration.from_dict(
            {
                "generator_config": {
                    "name": "VelocityGridSearch",
                    "min_vx": meanvx - 5,
                    "max_vx": meanvx + 5,
                    "min_vy": meanvy - 5,
                    "max_vy": meanvy + 5,
                    "vx_steps": 40,
                    "vy_steps": 40,
                },
                "num_obs": 1,
                "do_mask": False,
                "do_clustering": True,
                "do_stamp_filter": False,
            }
        )
        wu = ic.toWorkUnit(search_config)
        repr_wu = reproject_work_unit(wu, opt_wcs, parallelize=False)
        results = SearchRunner().run_search_from_work_unit(repr_wu)

        # Compare results and validate
        # - add in pixel velocities because realistic searches rarely
        #   find good pixel location match
        # - due to that, we also can't rely that we'll get a good match on
        #   any particular catalog realization. We iterate over all of them
        #   and find the best matching results in each realization.
        #   From all realizations find the one that matches the best.
        #   Select that realization and that best match for comparison.
        cats = obj_cat.mock(t=timestamps, wcs=[opt_wcs] * self.n_imgs)
        for cat in cats:
            cat["vx"] = meanvx
            cat["vy"] = meanvy

        dists = np.array([self.xmatch_best(cat, results)[1] for cat in cats])
        min_dist_within_realization = dists.min(axis=0)
        min_dist_across_realizations = dists.min()

        best_realization = dists.min(axis=1) == min_dist_across_realizations
        best_realization_idx = np.where(best_realization == True)[0][0]

        best_cat = cats[best_realization_idx]
        best_res = results[dists[best_realization_idx] == min_dist_across_realizations]

        self.assertGreaterEqual(len(results), 1)
        self.assertResultValuesWithinSpec(best_cat, best_res, 10)


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
