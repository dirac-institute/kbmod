"""Test some of the functions needed for running the search."""

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time

import unittest

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.reprojection_utils import fit_barycentric_wcs
from kbmod.results import Results
from kbmod.run_search import append_positions_to_results
from kbmod.search import *
from kbmod.wcs_utils import make_fake_wcs
from kbmod.work_unit import WorkUnit


class test_run_search(unittest.TestCase):
    def test_append_positions_to_results_global(self):
        # Create a fake WorkUnit with 20 times, a completely random ImageStack,
        # and no trajectories.
        num_times = 20
        width = 800
        height = 600
        t0 = 60676.0
        helio_dist = 500.0

        fake_times = create_fake_times(num_times, t0=t0)
        fake_ds = FakeDataSet(width, height, fake_times)

        # Create a global fake WCS,  one for each time (slightly shifted), and the EBD information.
        global_wcs = make_fake_wcs(20.0, 0.0, 800, 600, deg_per_pixel=0.5 / 3600.0)

        per_image_wcs = []
        for idx in range(num_times):
            # Each WCS is slight shifted from the global one.
            curr = make_fake_wcs(
                20.001 + idx / 1000.0, 0.001 + idx / 1000.0, 800, 600, deg_per_pixel=0.5 / 3600.0
            )
            per_image_wcs.append(curr)

        ebd_wcs, geo_dist = fit_barycentric_wcs(
            global_wcs,
            width,
            height,
            helio_dist,
            Time(t0, format="mjd"),
            EarthLocation.of_site("ctio"),
        )

        # Create the fake WorkUnit with this information.
        org_image_meta = Table(
            {
                "ebd_wcs": np.array([ebd_wcs] * num_times),
                "geocentric_distance": np.array([geo_dist] * num_times),
                "per_image_wcs": np.array(per_image_wcs),
            }
        )
        fake_wu = WorkUnit(
            im_stack=fake_ds.stack,
            config=SearchConfiguration(),
            wcs=ebd_wcs,
            reprojected=True,
            reprojection_frame="ebd",
            per_image_indices=[i for i in range(num_times)],
            heliocentric_distance=helio_dist,
            obstimes=fake_times,
            org_image_meta=org_image_meta,
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

        append_positions_to_results(fake_wu, results)

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

        # The per-image RA should exist, be close to (but not the same as)
        # the global RA for all observations.
        self.assertEqual(len(results["img_ra"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_ra"][i]), num_times)
            ra_diffs = np.abs(results["img_ra"][i] - results["global_ra"][i])
            self.assertTrue(np.all(ra_diffs > 0.0))
            self.assertTrue(np.all(ra_diffs < 1.0))

        # The per-image dec should exist, be close to (but not the same as)
        # the global dec for all observations.
        self.assertEqual(len(results["img_dec"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_dec"][i]), num_times)
            dec_diffs = np.abs(results["img_dec"][i] - results["global_dec"][i])
            self.assertTrue(np.all(dec_diffs > 0.0))
            self.assertTrue(np.all(dec_diffs < 1.0))

        # The per-image x should exist and be within some delta of the global predicted x.
        self.assertEqual(len(results["pred_x"]), 3)
        self.assertEqual(len(results["img_x"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_x"][i]), num_times)
            self.assertEqual(len(results["pred_x"][i]), num_times)
            x_diffs = np.abs(results["img_x"][i] - results["pred_x"][i])
            self.assertTrue(np.all(x_diffs > 0.0))
            self.assertTrue(np.all(x_diffs < 1000.0))

        # The per-image y should exist and be within some delta of the global predicted y.
        self.assertEqual(len(results["pred_y"]), 3)
        self.assertEqual(len(results["img_y"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_y"][i]), num_times)
            self.assertEqual(len(results["pred_y"][i]), num_times)
            y_diffs = np.abs(results["img_y"][i] - results["pred_y"][i])
            self.assertTrue(np.all(y_diffs > 0.0))
            self.assertTrue(np.all(y_diffs < 1000.0))

    def test_append_positions_to_results_no_global(self):
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

        append_positions_to_results(fake_wu, results)

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

        # The per-image x should exist and the same as global predicted x.
        self.assertEqual(len(results["pred_x"]), 3)
        self.assertEqual(len(results["img_x"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_x"][i]), num_times)
            self.assertEqual(len(results["pred_x"][i]), num_times)
            self.assertTrue(np.allclose(results["img_x"][i], results["pred_x"][i]))

        # The per-image y should exist and the same as global predicted y.
        self.assertEqual(len(results["pred_y"]), 3)
        self.assertEqual(len(results["img_y"]), 3)
        for i in range(3):
            self.assertEqual(len(results["img_y"][i]), num_times)
            self.assertEqual(len(results["pred_y"][i]), num_times)
            self.assertTrue(np.allclose(results["img_y"][i], results["pred_y"][i]))


if __name__ == "__main__":
    unittest.main()
