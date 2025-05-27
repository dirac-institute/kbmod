"""Test some of the functions needed for running the search."""

from astropy.coordinates import EarthLocation
from astropy.table import Table
from astropy.time import Time

import logging
import unittest

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import create_fake_times, FakeDataSet
from kbmod.reprojection_utils import fit_barycentric_wcs
from kbmod.results import Results
from kbmod.run_search import (
    append_positions_to_results,
    configure_kb_search_stack,
    SearchRunner,
)
from kbmod.search import *
from kbmod.trajectory_generator import VelocityGridSearch
from kbmod.wcs_utils import make_fake_wcs
from kbmod.work_unit import WorkUnit


class test_run_search(unittest.TestCase):
    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_run_search_bad_config(self):
        """Test cases where the search configuration is bad."""
        num_times = 20
        width = 150
        height = 100

        fake_times = create_fake_times(num_times, t0=60676.0)
        fake_ds = FakeDataSet(width, height, fake_times)

        runner = SearchRunner()

        # Turn off the warnings for this test.
        logging.disable(logging.CRITICAL)

        # Bad results_per_pixel.
        config = SearchConfiguration()
        config.set("results_per_pixel", -1)
        self.assertRaises(ValueError, runner.run_search, config, fake_ds.stack_py)

        # Bad search bounds.
        config = SearchConfiguration()
        config.set("x_pixel_bounds", [20, 10])
        self.assertRaises(ValueError, runner.run_search, config, fake_ds.stack_py)

        config = SearchConfiguration()
        config.set("y_pixel_bounds", [20, 10])
        self.assertRaises(ValueError, runner.run_search, config, fake_ds.stack_py)

        # Re-enable warnings.
        logging.disable(logging.NOTSET)

    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_run_search_auto_config(self):
        """Test that if we set num_obs to high, we down scale it."""
        num_times = 10
        width = 15
        height = 10

        fake_times = create_fake_times(num_times, t0=60676.0)
        fake_ds = FakeDataSet(width, height, fake_times)
        runner = SearchRunner()

        config = SearchConfiguration()
        config.set("num_obs", 21)

        # We also set the generator config so we do not get warnings about
        # a lack of ecliptic angles.
        generator_config = {
            "name": "EclipticCenteredSearch",
            "velocities": [92.0, 526.0, 257],
            "angles": [-np.pi / 15, np.pi / 15, 129],
            "angle_units": "radian",
            "velocity_units": "pix / d",
            "given_ecliptic": 0.0,
        }
        config.set("generator_config", generator_config)

        _ = runner.run_search(config, fake_ds.stack_py)
        self.assertEqual(config["num_obs"], 10)

    def test_load_and_filter_results(self):
        num_times = 50
        width = 20
        height = 5
        num_trjs = 10

        # Create a series of (unmoving) fake Trajectories to check.
        trjs = [Trajectory(x=i, y=0, vx=0.0, vy=0.0, obs_count=num_times, lh=100.0) for i in range(num_trjs)]

        # Create a fake dataset.
        fake_times = create_fake_times(num_times, t0=60676.0)
        fake_ds = FakeDataSet(width, height, fake_times)

        # Trajectory x is given x outliers.
        for i in range(num_times):
            sci = fake_ds.stack_py.sci[i]
            for x, trj in enumerate(trjs):
                if i < 2 * x:
                    sci[trj.y, trj.x] = 2000.0
                else:
                    sci[trj.y, trj.x] = 10.0

        # Set up the search object.  We turn off near duplicate filtering because
        # the trajectories are all near duplicates.
        config = SearchConfiguration()
        config.set("num_obs", 39)
        config.set("lh_level", 1.0)
        config.set("sigmaG_filter", True)
        config.set("sigmaG_lims", [10, 90])
        config.set("near_dup_thresh", None)

        search = StackSearch(
            fake_ds.stack_py.sci,
            fake_ds.stack_py.var,
            fake_ds.stack_py.psfs,
            fake_ds.stack_py.zeroed_times,
        )
        configure_kb_search_stack(search, config)

        # Try extracting before we have inserted any results. We should get an empty Results.
        runner = SearchRunner()
        results_empty = runner.load_and_filter_results(search, config, batch_size=10)
        self.assertEqual(len(results_empty), 0)

        # Insert the fake results.
        search.set_results(trjs)

        # Extract the (fake) results from the runner. We filter a bunch of
        # results that fall below 10 observations.
        runner = SearchRunner()
        results = runner.load_and_filter_results(search, config, batch_size=10)
        self.assertLess(len(results), 10)
        self.assertGreater(len(results), 2)

        # Reload with a different batch size and confirm we get the same results.
        results2 = runner.load_and_filter_results(search, config, batch_size=2)
        self.assertEqual(len(results2), len(results))
        results1 = runner.load_and_filter_results(search, config, batch_size=1)
        self.assertEqual(len(results1), len(results))

        # Re-extract without sigma-G filtering. We do not filter any results, but
        # still generate the psi and phi curves.
        config.set("sigmaG_filter", False)
        results = runner.load_and_filter_results(search, config)
        self.assertEqual(len(results), 10)
        self.assertTrue("psi_curve" in results.colnames)
        self.assertTrue("phi_curve" in results.colnames)

        # Turn off psi and phi curve generation and confirm we do not produce those.
        config.set("generate_psi_phi", False)
        results = runner.load_and_filter_results(search, config)
        self.assertEqual(len(results), 10)
        self.assertFalse("psi_curve" in results.colnames)
        self.assertFalse("phi_curve" in results.colnames)

    def test_append_positions_to_results_global(self):
        # Create a fake WorkUnit with 20 times, a completely random image stack,
        # and no trajectories.
        num_times = 20
        width = 800
        height = 600
        t0 = 60676.0
        bary_dist = 500.0

        fake_times = create_fake_times(num_times, t0=t0)
        fake_ds = FakeDataSet(width, height, fake_times)

        # Create a global fake WCS,  one for each time (slightly shifted), and the EBD information.
        global_wcs = make_fake_wcs(20.0, 0.0, 800, 600, deg_per_pixel=0.5 / 3600.0)

        per_image_wcs = []
        for idx in range(num_times):
            # Each WCS is slight shifted from the global one.
            curr = make_fake_wcs(
                20.001 + idx / 1000.0,
                0.001 + idx / 1000.0,
                800,
                600,
                deg_per_pixel=0.5 / 3600.0,
            )
            per_image_wcs.append(curr)

        ebd_wcs, geo_dist = fit_barycentric_wcs(
            global_wcs,
            width,
            height,
            bary_dist,
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
            im_stack=fake_ds.stack_py,
            config=SearchConfiguration(),
            wcs=ebd_wcs,
            reprojected=True,
            reprojection_frame="ebd",
            per_image_indices=[i for i in range(num_times)],
            barycentric_distance=bary_dist,
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
        # Create a fake WorkUnit with 20 times, a completely random image stack,
        # and no trajectories.
        num_times = 20
        fake_times = create_fake_times(num_times, t0=60676.0)
        fake_ds = FakeDataSet(800, 600, fake_times)

        # Append a global fake WCS and one for each time.
        all_wcs = []
        for idx in range(num_times):
            curr = make_fake_wcs(
                20.01 + idx / 100.0,
                0.01 + idx / 100.0,
                800,
                600,
                deg_per_pixel=0.5 / 3600.0,
            )
            all_wcs.append(curr)

        fake_wu = WorkUnit(
            im_stack=fake_ds.stack_py,
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

    def test_core_search_cpu(self):
        # Create a very small fake data set.
        num_times = 20
        width = 50
        height = 60
        fake_times = [float(i) / num_times for i in range(num_times)]
        fake_ds = FakeDataSet(width, height, fake_times, psf_val=0.01)

        # Create a Trajectory for the object and insert it into the image stack.
        trj = Trajectory(x=17, y=12, vx=21.0, vy=16.0, flux=250.0)
        fake_ds.insert_object(trj)

        # Use a small grid of search trajectories around the true velocity.
        trj_gen = VelocityGridSearch(5, 15.0, 27.0, 5, 10.0, 22.0)
        config = SearchConfiguration()
        config.set("cpu_only", True)

        # Run the core search algorithm and confirm we find the inserted fake.
        runner = SearchRunner()
        keep = runner.do_core_search(config, fake_ds.stack_py, trj_gen)

        self.assertGreater(len(keep), 0)
        self.assertEqual(keep["x"][0], 17)
        self.assertEqual(keep["y"][0], 12)
        self.assertAlmostEqual(keep["vx"][0], 21.0)
        self.assertAlmostEqual(keep["vy"][0], 16.0)

    @unittest.skipIf(not kb_has_gpu(), "Skipping test (no GPU detected)")
    def test_core_search_gpu(self):
        # Create a very small fake data set.
        num_times = 20
        width = 50
        height = 60
        fake_times = [float(i) / num_times for i in range(num_times)]
        fake_ds = FakeDataSet(width, height, fake_times, psf_val=0.01)

        # Create a Trajectory for the object and insert it into the image stack.
        trj = Trajectory(x=17, y=12, vx=21.0, vy=16.0, flux=250.0)
        fake_ds.insert_object(trj)

        # Use a small grid of search trajectories around the true velocity.
        trj_gen = VelocityGridSearch(5, 15.0, 27.0, 5, 10.0, 22.0)
        config = SearchConfiguration()

        # Run the core search algorithm and confirm we find the inserted fake.
        runner = SearchRunner()
        keep = runner.do_core_search(config, fake_ds.stack_py, trj_gen)
        self.assertGreater(len(keep), 0)
        self.assertEqual(keep["x"][0], 17)
        self.assertEqual(keep["y"][0], 12)
        self.assertAlmostEqual(keep["vx"][0], 21.0)
        self.assertAlmostEqual(keep["vy"][0], 16.0)


if __name__ == "__main__":
    unittest.main()
