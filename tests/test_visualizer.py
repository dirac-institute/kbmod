import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from kbmod.analysis.plotting import plot_result_row
from kbmod.analysis.visualizer import Visualizer
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.results import Results
from kbmod.search import Trajectory


class TestVisualizerWithoutLegacyStamp(unittest.TestCase):
    def test_daily_coadds_uses_all_stamps(self):
        times = np.array([60000.0, 60000.1, 60001.0])
        images = ImageStackPy(
            times=times,
            sci=[np.zeros((5, 5)) for _ in times],
            var=[np.ones((5, 5)) for _ in times],
        )
        results = Results.from_trajectories([Trajectory(2, 2, 0.0, 0.0)])
        results.update_obs_valid(np.array([[True, False, True]]))
        results.table["all_stamps"] = [
            np.array([np.full((3, 3), 1.0), np.full((3, 3), 100.0), np.full((3, 3), 3.0)])
        ]
        results.table["num_days"] = [2]

        visualizer = Visualizer(images, results)
        with patch("kbmod.analysis.visualizer.plot_multiple_images") as plot_images:
            visualizer.plot_daily_coadds(0)

        plotted = plot_images.call_args.args[0]
        self.assertTrue(np.allclose(plotted[0], 4.0))

    def test_plot_result_row_falls_back_to_named_coadd(self):
        results = Results.from_trajectories([Trajectory(2, 2, 0.0, 0.0)])
        results.table["coadd_mean"] = [np.full((3, 3), 2.0)]

        with patch("kbmod.analysis.plotting.plot_image") as plot_image:
            figure = plot_result_row(results.table[0])

        self.assertTrue(np.allclose(plot_image.call_args.args[0], 2.0))
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
