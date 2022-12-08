import unittest

import numpy as np

from kbmod.search import *


class test_search(unittest.TestCase):
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

        # Select one pixel to mask in every other image.
        self.masked_x = 5
        self.masked_y = 6

        # create image set with single moving object
        self.imlist = []
        for i in range(self.imCount):
            time = i / self.imCount
            im = layered_image(
                str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p, i
            )
            im.add_object(
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
            )

            # Mask a pixel in half the images.
            if i % 2 == 0:
                mask = im.get_mask()
                mask.set_pixel(self.masked_x, self.masked_y, 1)
                im.set_mask(mask)
                im.apply_mask_flags(1, [])

            self.imlist.append(im)
        self.stack = image_stack(self.imlist)
        self.search = stack_search(self.stack)

    def test_coadd_gpu_parity(self):
        radius = 2
        width = 2 * radius + 1
        params = stamp_parameters()
        params.radius = radius
        params.do_filtering = False

        results = [self.trj, self.trj]
        goodIdx = [[1] * self.imCount for _ in range(2)]
        goodIdx[1][0] = 0
        goodIdx[1][3] = 0
        goodIdx[1][5] = 0
        res_trjs = [trj_result(self.trj, goodIdx[0]), trj_result(self.trj, goodIdx[1])]

        # Check the summed stamps. Note summed stamp does not use goodIdx.
        params.stamp_type = StampType.STAMP_SUM
        stamps_old = self.search.summed_sci_stamps(res_trjs, radius)
        stamps_new = self.search.gpu_coadded_stamps(results, params)
        for r in range(2):
            for x in range(width):
                for y in range(width):
                    self.assertAlmostEqual(
                        stamps_old[r].get_pixel(x, y), stamps_new[r].get_pixel(x, y), delta=1e-5
                    )

        # Check the mean stamps.
        params.stamp_type = StampType.STAMP_MEAN
        stamps_old = self.search.mean_sci_stamps(res_trjs, radius)
        stamps_new = self.search.gpu_coadded_stamps(res_trjs, params)
        for r in range(2):
            for x in range(width):
                for y in range(width):
                    self.assertAlmostEqual(
                        stamps_old[r].get_pixel(x, y), stamps_new[r].get_pixel(x, y), delta=1e-5
                    )

        # Check the median stamps.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps_old = self.search.median_sci_stamps(res_trjs, radius)
        stamps_new = self.search.gpu_coadded_stamps(res_trjs, params)
        for r in range(2):
            for x in range(width):
                for y in range(width):
                    self.assertAlmostEqual(
                        stamps_old[r].get_pixel(x, y), stamps_new[r].get_pixel(x, y), delta=1e-5
                    )


if __name__ == "__main__":
    unittest.main()
