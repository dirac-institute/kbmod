import unittest

import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.filters.brightness_filters import extract_sci_var_curves
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
        expected_var_curves2 = np.array([[0, 3.6, 7.2, 10.8], [1.6, 5.7, 1.0, 1.0]])
        self.assertTrue(np.allclose(sci_curves, expected_sci_curves2, equal_nan=True))
        self.assertTrue(np.allclose(var_curves, expected_var_curves2, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
