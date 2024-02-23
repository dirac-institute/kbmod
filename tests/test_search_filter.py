import unittest

import numpy as np

from kbmod.candidate_generator import KBMODV1Search
from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.search import *
from kbmod.trajectory_utils import make_trajectory


class test_search_filter(unittest.TestCase):
    def setUp(self):
        # test pass thresholds
        self.pixel_error = 0
        self.velocity_error = 0.05
        self.flux_error = 0.15

        # image properties
        self.img_count = 20
        self.dim_x = 80
        self.dim_y = 60

        # object properties
        self.object_flux = 250.0
        self.start_x = 17
        self.start_y = 12
        self.vxel = 21.0
        self.vyel = 16.0

        # create a Trajectory for the object
        self.trj = make_trajectory(self.start_x, self.start_y, self.vxel, self.vyel, flux=self.object_flux)

        # search parameters
        self.angle_steps = 150
        self.velocity_steps = 150
        self.min_angle = 0.0
        self.max_angle = 1.5
        self.min_vel = 5.0
        self.max_vel = 40.0

        # Filtering parameters for the search.
        self.sigmaG_lims = np.array([0.25, 0.75])
        self.sigmaG_coeff = 0.7413
        self.lh_level = 10.0

        # create image set with single moving object
        fake_times = [i / self.img_count for i in range(self.img_count)]
        fake_ds = FakeDataSet(
            self.dim_x,
            self.dim_y,
            fake_times,
            noise_level=2.0,
            psf_val=1.0,
            use_seed=True,
        )
        fake_ds.insert_object(self.trj)

        self.search = StackSearch(fake_ds.stack)
        self.search.enable_gpu_sigmag_filter(self.sigmaG_lims, self.sigmaG_coeff, self.lh_level)
        strategy = KBMODV1Search(
            self.velocity_steps,
            self.min_vel,
            self.max_vel,
            self.angle_steps,
            self.min_angle,
            self.max_angle,
        )
        candidates = strategy.get_candidate_trajectories()
        self.search.search(candidates, int(self.img_count / 2))

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_results(self):
        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.vx / self.vxel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.vy / self.vyel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)


if __name__ == "__main__":
    unittest.main()
