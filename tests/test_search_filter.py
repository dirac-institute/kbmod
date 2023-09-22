import unittest

import numpy as np

from kbmod.fake_data_creator import add_fake_object
from kbmod.search import *


class test_search_filter(unittest.TestCase):
    def setUp(self):
        # test pass thresholds
        self.pixel_error = 0
        self.velocity_error = 0.05
        self.flux_error = 0.15

        # image properties
        self.imCount = 20
        self.dim_x = 80
        self.dim_y = 60
        self.noise_level = 4.0
        self.variance = self.noise_level**2
        self.p = psf(1.0)

        # object properties
        self.object_flux = 250.0
        self.start_x = 17
        self.start_y = 12
        self.x_vel = 21.0
        self.y_vel = 16.0

        # create a trajectory for the object
        self.trj = trajectory()
        self.trj.x = self.start_x
        self.trj.y = self.start_y
        self.trj.x_v = self.x_vel
        self.trj.y_v = self.y_vel

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
        self.imlist = []
        for i in range(self.imCount):
            time = i / self.imCount
            im = layered_image(
                str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i
            )
            add_fake_object(
                im,
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
                self.p,
            )
            self.imlist.append(im)
        self.stack = image_stack(self.imlist)

        self.search = stack_search(self.stack)
        self.search.enable_gpu_sigmag_filter(self.sigmaG_lims, self.sigmaG_coeff, self.lh_level)
        self.search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.imCount / 2),
        )

    def test_results(self):
        results = self.search.get_results(0, 10)
        best = results[0]
        self.assertAlmostEqual(best.x, self.start_x, delta=self.pixel_error)
        self.assertAlmostEqual(best.y, self.start_y, delta=self.pixel_error)
        self.assertAlmostEqual(best.x_v / self.x_vel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.y_v / self.y_vel, 1, delta=self.velocity_error)
        self.assertAlmostEqual(best.flux / self.object_flux, 1, delta=self.flux_error)


if __name__ == "__main__":
    unittest.main()
