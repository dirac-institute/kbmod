import unittest

import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.filters.brightness_filters import (
    apply_brightness_search_filter,
    extract_sci_var_curves,
    local_search_brightness,
    score_brightness_candidates,
)

from kbmod.results import Results
from kbmod.search import Trajectory


class TestBrightnessFilters(unittest.TestCase):
    def test_extract_sci_and_var_curves(self):
        height = 6
        width = 5
        num_times = 4

        times = np.arange(num_times)
        sci = np.arange(height * width * num_times).reshape((num_times, height, width))
        var = 0.1 * np.arange(height * width * num_times).reshape((num_times, height, width))
        im_stack = ImageStackPy(times, sci, var)

        trjs = [Trajectory(x=0, y=0, vx=1, vy=1), Trajectory(x=1, y=3, vx=1, vy=2)]
        results = Results.from_trajectories(trjs, track_filtered=False)
        self.assertFalse("sci_curve" in results.colnames)
        self.assertFalse("var_curve" in results.colnames)

        expected_sci_curves = np.array([[0, 36, 72, 108], [16, 57, np.nan, np.nan]])
        expected_var_curves = np.array([[0, 3.6, 7.2, 10.8], [1.6, 5.7, np.nan, np.nan]])

        sci_curves, var_curves = extract_sci_var_curves(results, im_stack, append=False)
        self.assertFalse("sci_curve" in results.colnames)
        self.assertFalse("var_curve" in results.colnames)
        self.assertTrue(np.allclose(sci_curves, expected_sci_curves, equal_nan=True))
        self.assertTrue(np.allclose(var_curves, expected_var_curves, equal_nan=True))

        _, _ = extract_sci_var_curves(results, im_stack, append=True)
        self.assertTrue("sci_curve" in results.colnames)
        self.assertTrue("var_curve" in results.colnames)
        self.assertTrue(np.allclose(results["sci_curve"], expected_sci_curves, equal_nan=True))
        self.assertTrue(np.allclose(results["var_curve"], expected_var_curves, equal_nan=True))

        sci_curves, var_curves = extract_sci_var_curves(results, im_stack, keep_nans=False, append=False)
        expected_sci_curves2 = np.array([[0, 36, 72, 108], [16, 57, 0.0, 0.0]])
        expected_var_curves2 = np.array([[0, 3.6, 7.2, 10.8], [1.6, 5.7, 1e10, 1e10]])

        self.assertTrue(np.allclose(sci_curves, expected_sci_curves2, equal_nan=True))
        self.assertTrue(np.allclose(var_curves, expected_var_curves2, equal_nan=True))

    def test_score_brightness_candidates(self):
        num_times = 10
        height = 40
        width = 50
        times = np.arange(num_times)

        # Create a fake data set a few fake objects with different fluxes.
        ds = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)
        ds.insert_random_object(flux=1)
        ds.insert_random_object(flux=20)
        ds.insert_random_object(flux=50)
        results = Results.from_trajectories(ds.trajectories, track_filtered=False)

        # Extract the science and variance curves.
        sci_curves, var_curves = extract_sci_var_curves(results, ds.stack_py, append=False)

        # Score the brightness candidates.
        brightness_candidates = np.array([0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0])
        scores = score_brightness_candidates(sci_curves, var_curves, brightness_candidates)
        self.assertEqual(scores.shape, (len(results), len(brightness_candidates)))

        # Check that the best score matches the true brightness of the object.
        best_idx = np.argmin(scores, axis=1)
        self.assertTrue(np.array_equal(best_idx, [0, 2, 4]))

    def test_local_search_brightness(self):
        num_times = 10
        height = 40
        width = 50
        times = np.arange(num_times)

        # Create a fake data set a few fake objects with different fluxes.
        ds = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)
        ds.insert_random_object(flux=1)
        ds.insert_random_object(flux=20)
        ds.insert_random_object(flux=50)
        results = Results.from_trajectories(ds.trajectories, track_filtered=False)

        # Extract the science and variance curves.
        sci_curves, var_curves = extract_sci_var_curves(results, ds.stack_py, append=False)

        # Score the brightness offsets. All of the objects should best match with their given flux.
        offsets = [0.5, 1.0, 1.5]
        best_brightness, best_idx = local_search_brightness(sci_curves, var_curves, offsets=offsets)
        self.assertEqual(best_brightness.shape, (len(results),))
        self.assertEqual(best_idx.shape, (len(results),))
        self.assertTrue(np.array_equal(best_idx, [1, 1, 1]))

    def test_apply_brightness_search_filter(self):
        num_times = 10
        height = 40
        width = 50
        times = np.arange(num_times)

        # Create a fake data set a few fake objects with different fluxes.
        ds = FakeDataSet(width=width, height=height, times=times, use_seed=11, psf_val=1e-6)
        ds.insert_random_object(flux=1)
        ds.insert_random_object(flux=20)
        ds.insert_random_object(flux=50)
        results = Results.from_trajectories(ds.trajectories, track_filtered=False)
        self.assertFalse("sci_curve" in results.colnames)
        self.assertFalse("var_curve" in results.colnames)

        # Filtering should not remove any of the objects.
        apply_brightness_search_filter(results, ds.stack_py, save_curves=True)
        self.assertTrue("sci_curve" in results.colnames)
        self.assertTrue("var_curve" in results.colnames)
        self.assertEqual(len(results), 3)

        # Make result 1's flux a massive overestimate and confirm that it gets filtered out.
        results["flux"][1] = 1000.0
        apply_brightness_search_filter(results, ds.stack_py)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
