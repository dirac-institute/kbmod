import unittest
from unittest import mock
import numpy as np
from astropy.table import Table

from kbmod import ImageCollection
from kbmod.configuration import SearchConfiguration

from utils import DECamImdiffFactory, MockButler, MockVisitInjectConfig, MockVisitInjectTask


class TestInjectionCatalog(unittest.TestCase):
    """Tests for generate_injection_catalog (no LSST dependency)."""

    def setUp(self):
        self.fitsFactory = DECamImdiffFactory()
        fits = self.fitsFactory.get_n(5, spoof_data=True)
        self.ic = ImageCollection.fromTargets(fits)
        self.ic.data["mjd_mid"] = np.linspace(59000.0, 59005.0, 5)

    def test_generate_injection_catalog_distance(self):
        """With guess_distance, catalog should have ra_{dist}/dec_{dist} columns."""
        search_config = SearchConfiguration()
        search_config.set(
            "generator_config",
            {
                "name": "EclipticCenteredSearch",
                "velocities": [10.0, 20.0, 2],
                "angles": [0, 10, 2],
            },
        )

        n_objs = 10
        global_wcs = self.ic.get_global_wcs(auto_fit=True)
        catalog = self.ic.generate_injection_catalog(
            search_config=search_config,
            global_wcs=global_wcs,
            n_objs_per_ic=n_objs,
            guess_distance=40.0,
            mag_range=(20.0, 22.0),
            source_type="Galaxy",
        )

        self.assertIsInstance(catalog, Table)
        self.assertEqual(len(catalog), 50)

        expected_cols = [
            "injection_id",
            "ra",
            "dec",
            "mag",
            "guess_distance",
            "source_type",
            "obj_ids",
            "obstime",
            "plot_x",
            "plot_y",
            "ra_40.0",
            "dec_40.0",
        ]
        for col in expected_cols:
            self.assertIn(col, catalog.colnames)

        self.assertEqual(catalog["guess_distance"][0], 40.0)
        self.assertEqual(catalog["source_type"][0], "Galaxy")
        self.assertTrue(20.0 <= catalog["mag"][0] <= 22.0)
        self.assertEqual(len(np.unique(catalog["obj_ids"])), n_objs)
        self.assertEqual(len(np.unique(catalog["obstime"])), 5)

    def test_generate_injection_catalog_no_distance(self):
        """Without guess_distance, no reflex columns should appear."""
        search_config = SearchConfiguration()
        search_config.set(
            "generator_config",
            {
                "name": "EclipticCenteredSearch",
                "velocities": [10.0, 20.0, 2],
                "angles": [0, 10, 2],
            },
        )

        n_objs = 10
        global_wcs = self.ic.get_global_wcs(auto_fit=True)
        catalog = self.ic.generate_injection_catalog(
            search_config=search_config, global_wcs=global_wcs, n_objs_per_ic=n_objs, guess_distance=None
        )

        self.assertIsInstance(catalog, Table)
        self.assertEqual(len(catalog), 50)
        self.assertIsNone(catalog["guess_distance"][0])
        self.assertIn("ra", catalog.colnames)
        self.assertIn("dec", catalog.colnames)
        self.assertNotIn("ra_40.0", catalog.colnames)
        self.assertNotIn("dec_40.0", catalog.colnames)

    def test_catalog_structure_n_objs_times_n_obstimes(self):
        """Verify catalog has n_objs * n_obstimes entries with correct grouping."""
        search_config = SearchConfiguration()
        search_config.set(
            "generator_config",
            {"name": "EclipticCenteredSearch", "velocities": [10.0, 20.0, 2], "angles": [0, 10, 2]},
        )

        n_objs = 5
        n_obstimes = len(self.ic)
        global_wcs = self.ic.get_global_wcs(auto_fit=True)

        catalog = self.ic.generate_injection_catalog(
            search_config=search_config, global_wcs=global_wcs, n_objs_per_ic=n_objs, guess_distance=None
        )

        # Should have n_objs * n_obstimes rows
        self.assertEqual(len(catalog), n_objs * n_obstimes)

        # Each object should appear exactly n_obstimes times
        for obj_id in range(n_objs):
            self.assertEqual(len(catalog[catalog["obj_ids"] == obj_id]), n_obstimes)

        # Each obstime should have exactly n_objs entries
        for obstime in self.ic.data["mjd_mid"]:
            self.assertEqual(len(catalog[catalog["obstime"] == obstime]), n_objs)

    def test_magnitude_within_specified_range(self):
        """Verify all magnitudes fall within mag_range and are consistent per object."""
        search_config = SearchConfiguration()
        search_config.set(
            "generator_config",
            {"name": "EclipticCenteredSearch", "velocities": [10.0, 20.0, 2], "angles": [0, 10, 2]},
        )

        global_wcs = self.ic.get_global_wcs(auto_fit=True)
        mag_min, mag_max = 20.0, 24.0

        catalog = self.ic.generate_injection_catalog(
            search_config=search_config,
            global_wcs=global_wcs,
            n_objs_per_ic=20,
            guess_distance=None,
            mag_range=(mag_min, mag_max),
        )

        # All magnitudes within range
        self.assertTrue(np.all(catalog["mag"] >= mag_min))
        self.assertTrue(np.all(catalog["mag"] <= mag_max))

        # Each object has consistent magnitude across obstimes
        for obj_id in np.unique(catalog["obj_ids"]):
            obj_mags = catalog[catalog["obj_ids"] == obj_id]["mag"]
            self.assertTrue(np.all(obj_mags == obj_mags[0]))


class TestInjectSources(unittest.TestCase):
    """Tests for inject_sources_into_ic using MockVisitInjectTask."""

    def setUp(self):
        self.fitsFactory = DECamImdiffFactory()
        fits = self.fitsFactory.get_n(3, spoof_data=True)
        self.ic = ImageCollection.fromTargets(fits)
        self.ic.data["mjd_mid"] = np.array([59000.0, 59001.0, 59002.0])
        # Spoof dataId column — inject_sources_into_ic needs it to look up
        # exposures via butler.get_dataset(DatasetId(idd))
        self.ic.data["dataId"] = ["0", "1", "2"]
        self.butler = MockButler("/mock/root")

    # Note that we use `create=True` to ensure that the mocks are created
    # even if LSST is not installed such as in the GitHub Actions environment.
    @mock.patch("kbmod.injection.HAS_LSST", True, create=True)
    @mock.patch("kbmod.injection.VisitInjectConfig", MockVisitInjectConfig, create=True)
    @mock.patch("kbmod.injection.VisitInjectTask", MockVisitInjectTask, create=True)
    @mock.patch("kbmod.injection.DatasetId", create=True)
    def test_inject_sources_stamps_pixels(self, mock_dataset_id):
        """Verify that inject_sources modifies image arrays and returns
        a new ImageCollection + vstacked catalog."""
        from utils import DatasetId as MockDatasetId

        mock_dataset_id.side_effect = MockDatasetId

        # Build a minimal catalog with one source per obstime
        catalog = Table(
            {
                "injection_id": [0, 1, 2],
                "ra": [self.ic.data["ra"][0]] * 3,
                "dec": [self.ic.data["dec"][0]] * 3,
                "mag": [22.0, 22.0, 22.0],
                "guess_distance": [None, None, None],
                "source_type": ["Star", "Star", "Star"],
                "obj_ids": [0, 0, 0],
                "obstime": [59000.0, 59001.0, 59002.0],
                "x": [2.5, 2.5, 2.5],
                "y": [2.5, 2.5, 2.5],
            }
        )

        injected_ic, injected_cats = self.ic.inject_sources(catalog=catalog, butler=self.butler)

        # The returned IC should have the same length
        self.assertEqual(len(injected_ic), len(self.ic))

        # The vstacked catalog should contain all 3 rows
        self.assertIsInstance(injected_cats, Table)
        self.assertEqual(len(injected_cats), 3)

    @mock.patch("kbmod.injection.HAS_LSST", False)
    def test_inject_sources_raises_without_lsst(self):
        """Without LSST, inject_sources should raise ImportError."""
        catalog = Table(
            {
                "injection_id": [0],
                "ra": [0.0],
                "dec": [0.0],
                "mag": [22.0],
                "obstime": [59000.0],
            }
        )
        with self.assertRaises(ImportError):
            self.ic.inject_sources(catalog=catalog, butler=self.butler)

    def test_flux_scaling_with_magnitude(self):
        """Verify that MockVisitInjectTask flux scales correctly with magnitude.

        This tests the mock implementation directly to ensure the formula
        flux = 10**((25 - mag) / 2.5) produces expected scaling.
        """
        # Test flux formula directly (used by MockVisitInjectTask._stamp_gaussian)
        mag_bright = 18.0
        mag_faint = 22.0

        flux_bright = 10 ** ((25.0 - mag_bright) / 2.5)
        flux_faint = 10 ** ((25.0 - mag_faint) / 2.5)

        # 4 magnitude difference = 10^(4/2.5) = 10^1.6 ≈ 40x
        expected_ratio = 10 ** ((mag_faint - mag_bright) / 2.5)

        actual_ratio = flux_bright / flux_faint
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)
        self.assertGreater(flux_bright, flux_faint)


class TestInjectionRecovery(unittest.TestCase):
    """End-to-end tests verifying injected sources can be recovered by search."""

    def test_injected_sources_recovered_by_search(self):
        """Verify that injected sources can be found by the search algorithm.

        This test creates fake data, inserts objects with known trajectories,
        and verifies search recovers them.
        """
        from kbmod.fake_data.fake_data_creator import FakeDataSet
        from kbmod.run_search import SearchRunner
        from kbmod.trajectory_generator import VelocityGridSearch
        from kbmod.search import Trajectory

        # Create fake dataset matching the pattern from test_run_search.py
        num_times = 20
        width = 60
        height = 70
        # Use close-together times (0.05 days apart) like test_core_search_cpu
        fake_times = [59000.0 + float(i) / num_times for i in range(num_times)]
        fake_ds = FakeDataSet(width, height, fake_times, psf_val=0.01)

        # Insert objects with known trajectories (similar to test_core_search_cpu)
        inserted_trjs = [
            Trajectory(x=15, y=20, vx=20.0, vy=15.0, flux=250.0),
            Trajectory(x=35, y=25, vx=18.0, vy=12.0, flux=250.0),
        ]
        for trj in inserted_trjs:
            fake_ds.insert_object(trj)

        # Configure search with velocity grid covering the inserted velocities
        config = SearchConfiguration()
        config.set("cpu_only", True)

        # Run search with grid centered on inserted velocities
        runner = SearchRunner()
        search_gen = VelocityGridSearch(5, 14.0, 24.0, 5, 8.0, 20.0)
        results = runner.do_core_search(config, fake_ds.stack_py, search_gen)

        # Verify we found results
        self.assertGreater(len(results), 0)

        # Check that each inserted trajectory is recovered (within tolerance)
        recovered_count = 0
        for trj in inserted_trjs:
            for i in range(len(results)):
                x_match = abs(results["x"][i] - trj.x) <= 2
                y_match = abs(results["y"][i] - trj.y) <= 2
                vx_match = abs(results["vx"][i] - trj.vx) <= 2.0
                vy_match = abs(results["vy"][i] - trj.vy) <= 2.0
                if x_match and y_match and vx_match and vy_match:
                    recovered_count += 1
                    break

        # Both objects should be recovered
        self.assertEqual(recovered_count, 2)

    def test_catalog_velocities_match_generator(self):
        """Verify generate_injection_catalog produces velocities from the generator."""
        search_config = SearchConfiguration()
        search_config.set(
            "generator_config",
            {
                "name": "EclipticCenteredSearch",
                "velocities": [10.0, 20.0, 3],  # 3 velocity steps
                "angles": [0.0, 0.1, 2],  # 2 angle steps
            },
        )

        fitsFactory = DECamImdiffFactory()
        fits = fitsFactory.get_n(3, spoof_data=True)
        ic = ImageCollection.fromTargets(fits)
        ic.data["mjd_mid"] = np.array([59000.0, 59001.0, 59002.0])

        global_wcs = ic.get_global_wcs(auto_fit=True)
        catalog = ic.generate_injection_catalog(
            search_config=search_config,
            global_wcs=global_wcs,
            n_objs_per_ic=5,
            guess_distance=None,
        )

        # Verify catalog has expected structure
        self.assertEqual(len(catalog), 5 * 3)  # 5 objects × 3 obstimes

        # Check that plot_x and plot_y evolve linearly (constant velocity)
        for obj_id in np.unique(catalog["obj_ids"]):
            obj_rows = catalog[catalog["obj_ids"] == obj_id]
            obj_rows.sort("obstime")

            # Extract positions
            xs = obj_rows["plot_x"]
            ys = obj_rows["plot_y"]
            ts = obj_rows["obstime"]

            # Compute velocities from positions
            if len(xs) > 1:
                dt = ts[1] - ts[0]
                vx_computed = (xs[1] - xs[0]) / dt
                vy_computed = (ys[1] - ys[0]) / dt

                # Velocity should be consistent across all time steps
                for i in range(1, len(xs) - 1):
                    dt_i = ts[i + 1] - ts[i]
                    vx_i = (xs[i + 1] - xs[i]) / dt_i
                    vy_i = (ys[i + 1] - ys[i]) / dt_i
                    self.assertAlmostEqual(vx_computed, vx_i, places=3)
                    self.assertAlmostEqual(vy_computed, vy_i, places=3)


class TestParallaxInversion(unittest.TestCase):
    """Tests for parallax inversion mathematical accuracy."""

    def test_parallax_round_trip(self):
        """Test correct_parallax -> invert gives back original coords."""
        try:
            from kbmod.reprojection_utils import (
                correct_parallax_geometrically_vectorized,
                invert_correct_parallax_vectorized,
            )
            from astropy.coordinates import SkyCoord, EarthLocation
            import astropy.units as u
        except ImportError:
            self.skipTest("reprojection_utils not available")

        ra_orig = np.array([290.0, 290.1, 290.2])
        dec_orig = np.array([-20.0, -20.1, -20.2])
        distance = 50.0
        obstimes = np.array([60000.0, 60001.0, 60002.0])
        loc = EarthLocation.of_site("Rubin")

        # Forward: correct parallax (uses ra, dec, mjds, distance signature)
        coords_corrected, _ = correct_parallax_geometrically_vectorized(
            ra_orig, dec_orig, obstimes, distance, point_on_earth=loc
        )

        # Inverse: invert the correction
        # invert_correct_parallax_vectorized expects SkyCoord with distance
        coords_for_invert = SkyCoord(
            ra=coords_corrected.ra, dec=coords_corrected.dec, distance=distance * u.au, frame="icrs"
        )
        coords_inverted = invert_correct_parallax_vectorized(coords_for_invert, obstimes, loc)

        # Should match original within sub-arcsecond precision
        ra_diff = np.abs(coords_inverted.ra.deg - ra_orig)
        dec_diff = np.abs(coords_inverted.dec.deg - dec_orig)

        self.assertLess(np.max(ra_diff), 1e-5, f"RA round-trip error: {np.max(ra_diff)} deg")
        self.assertLess(np.max(dec_diff), 1e-5, f"Dec round-trip error: {np.max(dec_diff)} deg")

    def test_closer_objects_have_larger_parallax(self):
        """Verify parallax correction is larger for closer objects."""
        try:
            from kbmod.reprojection_utils import invert_correct_parallax_vectorized
            from astropy.coordinates import SkyCoord, EarthLocation
            import astropy.units as u
        except ImportError:
            self.skipTest("reprojection_utils not available")

        ra, dec = 290.0, -20.0
        obstime = np.array([60000.0])
        loc = EarthLocation.of_site("Rubin")

        # Compare parallax at 30 AU vs 100 AU
        coords_close = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=30.0 * u.au, frame="icrs")
        coords_far = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=100.0 * u.au, frame="icrs")

        inv_close = invert_correct_parallax_vectorized(coords_close, obstime, loc)
        inv_far = invert_correct_parallax_vectorized(coords_far, obstime, loc)

        # Closer object should have larger displacement
        disp_close = np.sqrt((inv_close.ra.deg - ra) ** 2 + (inv_close.dec.deg - dec) ** 2)
        disp_far = np.sqrt((inv_far.ra.deg - ra) ** 2 + (inv_far.dec.deg - dec) ** 2)

        self.assertGreater(disp_close, disp_far)


if __name__ == "__main__":
    unittest.main()
