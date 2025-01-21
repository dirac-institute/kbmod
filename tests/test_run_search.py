"""Test some of the functions needed for running the search."""

import unittest

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.results import Results
from kbmod.run_search import append_ra_dec_to_results
from kbmod.search import *
from kbmod.wcs_utils import make_fake_wcs
from kbmod.work_unit import WorkUnit


class test_run_search(unittest.TestCase):
    def test_append_ra_dec_global(self):
        # Create a fake WorkUnit with 20 times, a completely random ImageStack,
        # and no trajectories.
        num_times = 20
        fake_times = create_fake_times(num_times, t0=60676.0)
        fake_ds = FakeDataSet(800, 600, fake_times)

        # Append a global fake WCS and one for each time.
        global_wcs = make_fake_wcs(20.0, 0.0, 800, 600, deg_per_pixel=0.5 / 3600.0)
        all_wcs = []
        for idx in range(num_times):
            curr = make_fake_wcs(
                20.01 + idx / 100.0, 0.01 + idx / 100.0, 800, 600, deg_per_pixel=0.5 / 3600.0
            )
            all_wcs.append(curr)

        fake_wu = WorkUnit(
            im_stack=fake_ds.stack,
            config=SearchConfiguration(),
            wcs=global_wcs,
            per_image_wcs=all_wcs,
            reprojected=True,
            reprojection_frame="ebd",
            per_image_indices=[i for i in range(num_times)],
            heliocentric_distance=np.full(num_times, 100.0),
            obstimes=fake_times,
        )

        # Create three fake trajectories in the bounds of the images. We don't
        # bother actually inserting them into the pixels.
        trjs = [
            Trajectory(x=5, y=10, vx=1, vy=1, flux=1000.0, lh=1000.0, obs_count=num_times),
            Trajectory(x=400, y=300, vx=-5, vy=-2, flux=1000.0, lh=1000.0, obs_count=num_times),
            Trajectory(x=100, y=500, vx=10, vy=-10, flux=1000.0, lh=1000.0, obs_count=num_times),
        ]
        results = Results.from_trajectories(trjs)
        self.assertEqual(len(results), 3)

        append_ra_dec_to_results(fake_wu, results)

        # The global RA should exist and be close to 20.0 for all observations.
        self.assertEqual(len(results["global_ra"]), 3)
        for i in range(3):
            self.assertEqual(len(results["global_ra"][i]), num_times)
            self.assertTrue(np.all(results["global_ra"][i] > 19.0))
            self.assertTrue(np.all(results["global_ra"][i] < 21.0))

        # The global Dec should exist and be close to 0.0 for all observations.
        self.assertEqual(len(results["global_dec"]), 3)
        for i in range(3):
            self.assertEqual(len(results["global_dec"][i]), num_times)
            self.assertTrue(np.all(results["global_dec"][i] > -1.0))
            self.assertTrue(np.all(results["global_dec"][i] < 1.0))

        # The per-image RA should exist, be close to 20.0 for all observations,
        # and be different from the global RA
        self.assertEqual(len(results["img_ra"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_ra"][i]), num_times)
            self.assertTrue(np.all(results["img_ra"][i] > 19.0))
            self.assertTrue(np.all(results["img_ra"][i] < 21.0))
            self.assertFalse(np.any(results["img_ra"][i] == results["global_ra"][i]))

        # The global Dec should exist and be close to 0.0 for all observations.
        self.assertEqual(len(results["img_dec"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_dec"][i]), num_times)
            self.assertTrue(np.all(results["img_dec"][i] > -1.0))
            self.assertTrue(np.all(results["img_dec"][i] < 1.0))
            self.assertFalse(np.any(results["img_dec"][i] == results["global_dec"][i]))

    def test_append_ra_dec_no_global(self):
        # Create a fake WorkUnit with 20 times, a completely random ImageStack,
        # and no trajectories.
        num_times = 20
        fake_times = create_fake_times(num_times, t0=60676.0)
        fake_ds = FakeDataSet(800, 600, fake_times)

        # Append a global fake WCS and one for each time.
        all_wcs = []
        for idx in range(num_times):
            curr = make_fake_wcs(
                20.01 + idx / 100.0, 0.01 + idx / 100.0, 800, 600, deg_per_pixel=0.5 / 3600.0
            )
            all_wcs.append(curr)

        fake_wu = WorkUnit(
            im_stack=fake_ds.stack,
            config=SearchConfiguration(),
            wcs=None,
            per_image_wcs=all_wcs,
            reprojected=False,
            per_image_indices=[i for i in range(num_times)],
            obstimes=fake_times,
        )

        # Create three fake trajectories in the bounds of the images. We don't
        # bother actually inserting them into the pixels.
        trjs = [
            Trajectory(x=5, y=10, vx=1, vy=1, flux=1000.0, lh=1000.0, obs_count=num_times),
            Trajectory(x=400, y=300, vx=-5, vy=-2, flux=1000.0, lh=1000.0, obs_count=num_times),
            Trajectory(x=100, y=500, vx=10, vy=-10, flux=1000.0, lh=1000.0, obs_count=num_times),
        ]
        results = Results.from_trajectories(trjs)
        self.assertEqual(len(results), 3)

        append_ra_dec_to_results(fake_wu, results)

        # The global RA and global dec should not exist.
        self.assertFalse("global_ra" in results.colnames)
        self.assertFalse("global_dec" in results.colnames)

        # The per-image RA should exist, be close to 20.0 for all observations.
        self.assertEqual(len(results["img_ra"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_ra"][i]), num_times)
            self.assertTrue(np.all(results["img_ra"][i] > 19.0))
            self.assertTrue(np.all(results["img_ra"][i] < 21.0))

        # The global Dec should exist and be close to 0.0 for all observations.
        self.assertEqual(len(results["img_dec"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_dec"][i]), num_times)
            self.assertTrue(np.all(results["img_dec"][i] > -1.0))
            self.assertTrue(np.all(results["img_dec"][i] < 1.0))


if __name__ == "__main__":
    unittest.main()
