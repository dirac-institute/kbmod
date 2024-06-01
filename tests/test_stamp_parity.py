"""
KBMOD provides a series of different wrapper functions for extracting
coadded stamps from trajectories. These tests confirm that the behavior
of the different approaches is consistent.
"""

import unittest

import numpy as np

from kbmod.fake_data.fake_data_creator import FakeDataSet
from kbmod.search import *


class test_search(unittest.TestCase):
    def setUp(self):
        # image properties
        self.img_count = 20
        self.dim_x = 80
        self.dim_y = 60

        # create a Trajectory for the object
        self.trj = Trajectory(17, 12, 21.0, 16.0, flux=250.0)

        # Select one pixel to mask in every other image.
        self.masked_x = 5
        self.masked_y = 6

        # create image set with single moving object
        fake_times = [i / self.img_count for i in range(self.img_count)]
        self.fake_ds = FakeDataSet(
            self.dim_x,
            self.dim_y,
            fake_times,
            noise_level=2.0,
            psf_val=1.0,
            use_seed=True,
        )
        self.fake_ds.insert_object(self.trj)

        # Mask a pixel in half the images.
        for i in range(self.img_count):
            if i % 2 == 0:
                img = self.fake_ds.stack.get_single_image(i)
                img.get_mask().set_pixel(self.masked_y, self.masked_x, 1)
                img.apply_mask(1)
        self.search = StackSearch(self.fake_ds.stack)

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_coadd_gpu_parity(self):
        radius = 2
        width = 2 * radius + 1
        params = StampParameters()
        params.radius = radius
        params.do_filtering = False

        results = [self.trj, self.trj]
        all_valid = [1] * self.img_count
        goodIdx = [[1] * self.img_count for _ in range(2)]
        goodIdx[1][0] = 0
        goodIdx[1][3] = 0
        goodIdx[1][5] = 0

        # Check the summed stamps. Note summed stamp does not use goodIdx.
        params.stamp_type = StampType.STAMP_SUM
        stamps_old = [
            StampCreator.get_summed_stamp(self.search.get_imagestack(), self.trj, radius, all_valid),
            StampCreator.get_summed_stamp(self.search.get_imagestack(), self.trj, radius, all_valid),
        ]
        stamps_gpu = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), results, [all_valid, all_valid], params, True
        )
        stamps_cpu = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), results, [all_valid, all_valid], params, False
        )
        for r in range(2):
            self.assertTrue(np.allclose(stamps_old[r].image, stamps_gpu[r].image, atol=1e-5))
            self.assertTrue(np.allclose(stamps_old[r].image, stamps_cpu[r].image, atol=1e-5))

        # Check the mean stamps.
        params.stamp_type = StampType.STAMP_MEAN
        stamps_old = [
            StampCreator.get_mean_stamp(self.search.get_imagestack(), self.trj, radius, goodIdx[0]),
            StampCreator.get_mean_stamp(self.search.get_imagestack(), self.trj, radius, goodIdx[1]),
        ]
        stamps_gpu = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), results, goodIdx, params, True
        )
        stamps_cpu = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), results, goodIdx, params, False
        )
        for r in range(2):
            self.assertTrue(np.allclose(stamps_old[r].image, stamps_gpu[r].image, atol=1e-5))
            self.assertTrue(np.allclose(stamps_old[r].image, stamps_cpu[r].image, atol=1e-5))

        # Check the median stamps.
        params.stamp_type = StampType.STAMP_MEDIAN
        stamps_old = [
            StampCreator.get_median_stamp(self.search.get_imagestack(), self.trj, radius, goodIdx[0]),
            StampCreator.get_median_stamp(self.search.get_imagestack(), self.trj, radius, goodIdx[1]),
        ]
        stamps_gpu = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), results, goodIdx, params, True
        )
        stamps_cpu = StampCreator.get_coadded_stamps(
            self.search.get_imagestack(), results, goodIdx, params, False
        )
        for r in range(2):
            self.assertTrue(np.allclose(stamps_old[r].image, stamps_gpu[r].image, 1e-5))
            self.assertTrue(np.allclose(stamps_old[r].image, stamps_cpu[r].image, 1e-5))


if __name__ == "__main__":
    unittest.main()
