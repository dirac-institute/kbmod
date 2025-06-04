import unittest

import numpy as np

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.filters.brightness_filters import (
    extract_sci_curves,
    extract_var_curves,
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

        sci_curves = extract_sci_curves(results, im_stack, append=True)
        expected_sci_curves = [[0, 36, 72, 108], [16, 57, np.nan, np.nan]]
        self.assertTrue("sci_curves" in results.colnames)
        self.assertEqual(sci_curves.shape, (len(trjs), num_times))
        self.assertTrue(np.allclose(sci_curves, expected_sci_curves, equal_nan=True))

        var_curves = extract_var_curves(results, im_stack, append=True)
        expected_var_curves = [[0, 3.6, 7.2, 10.8], [1.6, 5.7, np.nan, np.nan]]
        self.assertTrue("var_curves" in results.colnames)
        self.assertEqual(var_curves.shape, (len(trjs), num_times))
        self.assertTrue(np.allclose(var_curves, expected_var_curves, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
