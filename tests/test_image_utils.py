import numpy as np
import unittest

from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.search import Trajectory

from kbmod.image_utils import (
    create_stamps_from_image_stack,
    create_stamps_from_image_stack_xy,
)


class test_image_utils(unittest.TestCase):
    def test_create_stamps_from_image_stack(self):
        # Create a small fake data set for the tests.
        num_times = 10
        fake_times = create_fake_times(num_times, 57130.2, 1, 0.01, 1)
        fake_ds = FakeDataSet(
            25,  # width
            35,  # height
            fake_times,  # time stamps
            noise_level=1.0,  # noise level
            psf_val=0.5,  # psf value
            use_seed=True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        fake_ds.insert_object(trj)

        # Create stamps from the fake data set and Trajectory.
        stamps = create_stamps_from_image_stack(fake_ds.stack_py, trj, 1)
        self.assertEqual(len(stamps), num_times)
        for i in range(num_times):
            self.assertEqual(stamps[i].shape, (3, 3))

            # Compare to the (manually computed) trajectory location.
            xp = 8 + 2 * i
            yp = 7 + i
            if xp < 25 and yp < 35:
                center_val = fake_ds.stack_py.sci[i][yp, xp]
                self.assertAlmostEqual(center_val, stamps[i][1, 1])
            else:
                self.assertTrue(np.isnan(stamps[i][1, 1]))

        # Check that we can set use_indices to produce only some stamps.
        use_times = [False, True, False, True, True, False, False, False, True, False]
        stamps = create_stamps_from_image_stack(fake_ds.stack_py, trj, 1, to_include=use_times)
        self.assertEqual(len(stamps), np.count_nonzero(use_times))

        stamp_count = 0
        for i in range(num_times):
            if use_times[i]:
                self.assertEqual(stamps[stamp_count].shape, (3, 3))

                xp = 8 + 2 * i
                yp = 7 + i
                if xp < 25 and yp < 35:
                    center_val = fake_ds.stack_py.sci[i][yp, xp]
                    self.assertAlmostEqual(center_val, stamps[stamp_count][1, 1])
                else:
                    self.assertTrue(np.isnan(stamps[stamp_count][1, 1]))

                stamp_count += 1

    def test_create_stamps_from_image_stack_xy(self):
        # Create a small fake data set for the tests.
        num_times = 10
        fake_times = create_fake_times(num_times, 57130.2, 1, 0.01, 1)
        fake_ds = FakeDataSet(
            25,  # width
            35,  # height
            fake_times,  # time stamps
            noise_level=1.0,  # noise level
            psf_val=0.5,  # psf value
            use_seed=True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        trj = Trajectory(8, 7, 2.0, 1.0, flux=250.0)
        fake_ds.insert_object(trj)

        zeroed_times = np.array(fake_ds.stack_py.zeroed_times)
        xvals = (trj.x + trj.vx * zeroed_times + 0.5).astype(int)
        yvals = (trj.y + trj.vy * zeroed_times + 0.5).astype(int)
        stamps = create_stamps_from_image_stack_xy(fake_ds.stack_py, 1, xvals, yvals)
        self.assertEqual(len(stamps), num_times)
        for i in range(num_times):
            self.assertEqual(stamps[i].shape, (3, 3))

            # Compare to the (manually computed) trajectory location.
            xp = 8 + 2 * i
            yp = 7 + i
            if xp < 25 and yp < 35:
                center_val = fake_ds.stack_py.sci[i][yp, xp]
                self.assertAlmostEqual(center_val, stamps[i][1, 1])
            else:
                self.assertTrue(np.isnan(stamps[i][1, 1]))

        # Check that we can set use_indices to produce only some stamps.
        use_inds = np.array([1, 2, 3, 5, 6])
        stamps = create_stamps_from_image_stack_xy(fake_ds.stack_py, 1, xvals, yvals, to_include=use_inds)
        self.assertEqual(len(stamps), len(use_inds))

        for stamp_i, image_i in enumerate(use_inds):
            self.assertEqual(stamps[stamp_i].shape, (3, 3))

            xp = 8 + 2 * image_i
            yp = 7 + image_i
            if xp < 25 and yp < 35:
                center_val = fake_ds.stack_py.sci[image_i][yp, xp]
                self.assertAlmostEqual(center_val, stamps[stamp_i][1, 1])
            else:
                self.assertTrue(np.isnan(stamps[stamp_i][1, 1]))


if __name__ == "__main__":
    unittest.main()
