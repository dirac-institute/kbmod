import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
from astropy.table import Table, MaskedColumn
from astropy.time import Time

from kbmod import ImageCollection
from kbmod.configuration import SearchConfiguration
from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized
from kbmod.search import Trajectory
from kbmod.injection import generate_injection_catalog, inject_sources_into_ic, match_injection_results
from kbmod.results import Results
from kbmod.standardizers import ButlerStandardizer
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from utils import (
    DECamImdiffFactory,
    MockButler,
    MockVisitInjectConfig,
    MockVisitInjectTask,
    DatasetId,
    dafButler,
)


class TestInjectionCatalog(unittest.TestCase):
    """Tests for generate_injection_catalog (no LSST dependency)."""

    def setUp(self):
        self.fitsFactory = DECamImdiffFactory()
        fits = self.fitsFactory.get_n(5, spoof_data=True)
        self.ic = ImageCollection.fromTargets(fits)
        self.ic.data["mjd_mid"] = np.linspace(59000.0, 59005.0, 5)

    def test_generate_injection_catalog_distance(self):
        """With guess_distance, catalog should have ra_{dist}/dec_{dist} columns."""
        # Set up a search configuration for the search space to simulate.
        search_config = SearchConfiguration()
        search_config.set(
            "generator_config",
            {
                "name": "EclipticCenteredSearch",
                "velocities": [10.0, 20.0, 2],
                "angles": [0, 10, 2],
            },
        )

        # The number of objects to inject per image collection.
        n_objs = 10
        global_wcs = self.ic.get_global_wcs(auto_fit=True)
        # Perform our insertion.
        catalog = generate_injection_catalog(
            ic=self.ic,
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

        n_obstimes = len(self.ic)
        # Verify catalog has n_objs * n_obstimes entries
        self.assertEqual(len(catalog), n_objs * n_obstimes)

        # Each object should appear exactly n_obstimes times
        for obj_id in range(n_objs):
            self.assertEqual(len(catalog[catalog["obj_ids"] == obj_id]), n_obstimes)

        # Each obstime should have exactly n_objs entries
        for obstime in self.ic.data["mjd_mid"]:
            self.assertEqual(len(catalog[catalog["obstime"] == obstime]), n_objs)

        # All magnitudes within specified mag_range
        self.assertTrue(np.all(catalog["mag"] >= 20.0))
        self.assertTrue(np.all(catalog["mag"] <= 22.0))

        # Each object has consistent magnitude across obstimes
        for obj_id in np.unique(catalog["obj_ids"]):
            obj_mags = catalog[catalog["obj_ids"] == obj_id]["mag"]
            self.assertTrue(np.all(obj_mags == obj_mags[0]))

        # Verify that the (ra, dec) coordinates are indeed the inverse-parallax corrected
        # versions of the (ra_40.0, dec_40.0) straight-line trajectories.
        # We do this by applying the forward correction to (ra, dec) and asserting it yields (ra_40.0, dec_40.0)
        max_ra_diff_naive = np.max(np.abs(catalog["ra"] - catalog["ra_40.0"]))
        max_dec_diff_naive = np.max(np.abs(catalog["dec"] - catalog["dec_40.0"]))
        # Parallax should have moved the coordinates by a measurable amount (> 1e-7 deg)
        self.assertGreater(max_ra_diff_naive, 1e-7)
        self.assertGreater(max_dec_diff_naive, 1e-7)

        loc = self.ic.get_observatory()
        coords_forward, _ = correct_parallax_geometrically_vectorized(
            catalog["ra"], catalog["dec"], catalog["obstime"], 40.0, point_on_earth=loc
        )

        # The forward-corrected coordinates should exactly match the straight-line coords
        ra_diff = np.abs(coords_forward.ra.deg - catalog["ra_40.0"])
        dec_diff = np.abs(coords_forward.dec.deg - catalog["dec_40.0"])

        self.assertLess(np.max(ra_diff), 1e-6)
        self.assertLess(np.max(dec_diff), 1e-6)

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
        catalog = generate_injection_catalog(
            ic=self.ic,
            search_config=search_config,
            global_wcs=global_wcs,
            n_objs_per_ic=n_objs,
            guess_distance=None,
        )

        self.assertIsInstance(catalog, Table)
        self.assertEqual(len(catalog), 50)
        self.assertIsNone(catalog["guess_distance"][0])

        # Assert that only one column has the substring "ra" since no guess_distance was provided.
        ra_cols = [col for col in catalog.colnames if "ra" in col]
        self.assertEqual(len(ra_cols), 1)

        # Assert that only one column has the substring "dec" since no guess_distance was provided.
        dec_cols = [col for col in catalog.colnames if "dec" in col]
        self.assertEqual(len(dec_cols), 1)

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
        catalog = generate_injection_catalog(
            ic=ic,
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


class TestInjectSources(unittest.TestCase):
    """Tests for inject_sources_into_ic using MockVisitInjectTask."""

    def setUp(self):
        modules = mock.patch.dict("sys.modules", {"lsst.daf.butler": dafButler})
        modules.start()
        self.addCleanup(modules.stop)
        # Use use_header_dimensions=True so the mock exposure WCS is consistent
        # with the image bounds (RA/Dec -> pixel conversions land within image)
        self.butler = MockButler("/mock/root", use_header_dimensions=True)
        original_visit = self.butler.mock_visitinfo

        def visit_info(ref):
            visit = original_visit(ref)
            visit.date.toAstropy.return_value = Time(59000.0 + ref, format="mjd", scale="utc")
            return visit

        visits = mock.patch.object(self.butler, "mock_visitinfo", side_effect=visit_info)
        visits.start()
        self.addCleanup(visits.stop)
        self.ic = ImageCollection.fromStandardizers(
            [ButlerStandardizer(DatasetId(i, fill_metadata=True), butler=self.butler) for i in range(3)]
        )

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

        # Build a minimal catalog with one source per obstime.
        # Use the center RA/Dec of each exposure to ensure sources are within bounds.
        catalog = Table(
            {
                "injection_id": [0, 1, 2],
                "ra": list(self.ic.data["ra"][:3]),
                "dec": list(self.ic.data["dec"][:3]),
                "mag": [22.0, 22.0, 22.0],
                "guess_distance": [None, None, None],
                "source_type": ["Star", "Star", "Star"],
                "obj_ids": [0, 0, 0],
                "obstime": list(self.ic.data["mjd_mid"]),
                "plot_x": [2.5, 2.5, 2.5],
                "plot_y": [2.5, 2.5, 2.5],
            }
        )

        injected_ic, injected_cats = inject_sources_into_ic(self.ic, catalog=catalog, butler=self.butler)

        # The returned IC should have the same length
        self.assertEqual(len(injected_ic), len(self.ic))

        # The vstacked catalog should contain all 3 rows
        self.assertIsInstance(injected_cats, Table)
        self.assertEqual(len(injected_cats), 3)

        # Explicitly verify the array was modified in place and retained by the new standardizer.
        # The underlying array should have a visible peak from the Gaussian stamp.
        orig_stds = self.ic.get_standardizers(butler=self.butler)
        injected_stds = injected_ic.get_standardizers()

        for i in range(len(self.ic)):
            orig_array = list(orig_stds[i]["std"].standardizeScienceImage())[0]
            injected_array = injected_stds[i]["std"].exp.image.array

            # The injected array must have a strictly higher sum of pixel values
            # due to the positive Gaussian flux injection.
            self.assertGreater(np.sum(injected_array), np.sum(orig_array))

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
            inject_sources_into_ic(self.ic, catalog=catalog, butler=self.butler)

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


class TestInjectionTiming(unittest.TestCase):
    """Exercise production orchestration; only Rubin rendering is stubbed."""

    def setUp(self):
        def patch_context(patcher):
            value = patcher.start()
            self.addCleanup(patcher.stop)
            return value

        patch_context(mock.patch.dict("sys.modules", {"lsst.daf.butler": dafButler}))
        self.butler = MockButler("/timing/mock")
        original_visit = self.butler.mock_visitinfo

        def visit_info(ref):
            visit = original_visit(ref)
            visit.exposureTime = 120.0
            visit.date.toAstropy.return_value = Time("2025-05-02T01:00:37", scale="tai") + ref * 300 * u.s
            return visit

        patch_context(mock.patch.object(self.butler, "mock_visitinfo", side_effect=visit_info))
        self.ic = ImageCollection.fromStandardizers(
            [ButlerStandardizer(DatasetId(i, fill_metadata=True), butler=self.butler) for i in range(2)]
        )
        self.native = np.asarray(self.ic.data["mjd_mid"]).copy()
        ra, dec = list(self.ic.wcs)[0].all_pix2world(2, 2, 0)
        self.catalog = Table(
            {
                "injection_id": [0, 1],
                "obstime": self.native.copy(),
                "ra": [float(ra)] * 2,
                "dec": [float(dec)] * 2,
                "mag": [22.0] * 2,
                "obj_ids": ["object"] * 2,
                "source_type": ["Star"] * 2,
            }
        )
        patch_context(mock.patch("kbmod.injection.HAS_LSST", True))
        patch_context(mock.patch("kbmod.injection.DatasetId", DatasetId, create=True))
        patch_context(mock.patch("kbmod.injection.VisitInjectConfig", MockVisitInjectConfig, create=True))
        self.task_factory = patch_context(mock.patch("kbmod.injection.VisitInjectTask", create=True))
        self.task = self.task_factory.return_value
        self.task.run.side_effect = lambda injection_catalogs, input_exposure, **kw: SimpleNamespace(
            output_exposure=input_exposure, output_catalog=injection_catalogs
        )
        self.pixel_load = patch_context(
            mock.patch.object(self.butler, "mock_exposure", wraps=self.butler.mock_exposure)
        )

    def assert_preflight_failure(self, ic, catalog, message):
        with self.assertRaisesRegex(ValueError, message):
            inject_sources_into_ic(ic, catalog, self.butler)
        self.task_factory.assert_not_called()
        self.pixel_load.assert_not_called()

    def test_legacy_collection_fails_before_any_pixels(self):
        for cache in ("warm", "cold"):
            with self.subTest(cache=cache):
                # Put the error in the LAST row: no earlier exposure may be mutated.
                self.ic.data["mjd_mid"][1] = self.native[1] + 60.5 / 86400
                catalog = self.catalog.copy()
                catalog["obstime"] = self.ic.data["mjd_mid"].copy()
                if cache == "warm":
                    self.ic._standardizers[1]._metadata["mjd_mid"] = catalog["obstime"][1]
                else:
                    self.ic._standardizers = np.full((2,), None)
                stored = np.asarray(self.ic.data["mjd_mid"]).copy()
                self.assert_preflight_failure(self.ic, catalog, "timestamp mismatch.*row 1")
                np.testing.assert_array_equal(self.ic.data["mjd_mid"], stored)
                np.testing.assert_array_equal(catalog["obstime"], stored)

    def test_stale_precomputed_catalog(self):
        for rows in ([0, 1], [1]):
            with self.subTest(rows=rows):
                catalog = self.catalog.copy()
                catalog["obstime"][rows] += 60.5 / 86400
                self.assert_preflight_failure(self.ic, catalog, "unmatched obstime")

    def test_invalid_epochs(self):
        for target in ("collection", "catalog"):
            for value in (np.nan, np.inf, "masked"):
                with self.subTest(target=target, value=value):
                    ic = ImageCollection(self.ic.data.copy(), standardizers=self.ic._standardizers)
                    catalog = self.catalog.copy()
                    table, column = (ic.data, "mjd_mid") if target == "collection" else (catalog, "obstime")
                    if value == "masked":
                        table.replace_column(column, MaskedColumn(table[column], mask=[False, True]))
                    else:
                        table[column][1] = value
                    self.assert_preflight_failure(ic, catalog, "finite, unmasked")

    def test_ambiguous_catalog_epoch(self):
        self.ic.data["mjd_mid"][1] = self.native[0] + 0.0005 / 86400
        self.assert_preflight_failure(self.ic, self.catalog[:1], "ambiguous obstime")

    def test_corrected_catalog_roundoff_preserves_epochs_and_matches(self):
        # The second exposure intentionally has no catalog sources.
        catalog = self.catalog[:1].copy()
        catalog["obstime"] += 0.0001 / 86400
        stored = np.asarray(catalog["obstime"]).copy()
        rebuilt, truth = inject_sources_into_ic(self.ic, catalog, self.butler)
        self.assertEqual(self.task.run.call_count, 1)
        np.testing.assert_array_equal(rebuilt.data["mjd_mid"], self.native)
        np.testing.assert_array_equal(truth["obstime"], stored)
        np.testing.assert_array_equal(catalog["obstime"], stored)
        # Mock incidental metadata fields are objects, unlike real Butler values.
        for name in list(rebuilt.data.colnames):
            if rebuilt.data[name].dtype.kind == "O":
                rebuilt.data.remove_column(name)
        work = rebuilt.toWorkUnit(butler=self.butler)
        np.testing.assert_array_equal(work.get_all_obstimes(), self.native)
        result = Results(
            Table(
                {
                    "x": [2],
                    "y": [2],
                    "vx": [0.0],
                    "vy": [0.0],
                    "likelihood": [10.0],
                    "flux": [1.0],
                    "obs_count": [1],
                    "obs_valid": [[True, False]],
                }
            ),
            wcs=list(rebuilt.wcs)[0],
        )
        result.set_mjd_utc_mid(work.get_all_obstimes())
        match_injection_results(truth, result, min_obs=1)
        self.assertEqual(result.table["injected_sources"][0]["object"], [True, False])

    def test_empty_catalog_is_allowed(self):
        with self.assertWarnsRegex(UserWarning, "No objects were successfully rendered"):
            rebuilt, truth = inject_sources_into_ic(self.ic, self.catalog[:0], self.butler)
        self.task.run.assert_not_called()
        self.assertEqual(len(truth), 0)
        np.testing.assert_array_equal(rebuilt.data["mjd_mid"], self.native)

    def test_multiple_detectors_at_one_epoch(self):
        # Catalog epochs identify visits, so both detector images should receive
        # the same catalog rows without being treated as an ambiguous match.
        self.ic.data["mjd_mid"][1] = self.native[0]
        self.ic._standardizers[1]._metadata["mjd_mid"] = self.native[0]
        visit = self.butler.mock_visitinfo(0)
        with mock.patch.object(self.butler, "mock_visitinfo", return_value=visit):
            rebuilt, truth = inject_sources_into_ic(self.ic, self.catalog[:1], self.butler)
        self.assertEqual(self.task.run.call_count, 2)
        self.assertEqual(len(truth), 2)
        np.testing.assert_array_equal(rebuilt.data["mjd_mid"], [self.native[0]] * 2)

    def test_rebuilt_epoch_drift_is_rejected(self):
        # Fresh source and collection times agree; a stale cached standardizer
        # must not be allowed to change the returned collection's convention.
        self.ic._standardizers[1]._metadata["mjd_mid"] += 60.5 / 86400
        with self.assertRaisesRegex(ValueError, "Injection changed the ImageCollection midpoints"):
            inject_sources_into_ic(self.ic, self.catalog, self.butler)
        self.assertEqual(self.task.run.call_count, 2)
        np.testing.assert_array_equal(self.ic.data["mjd_mid"], self.native)


class TestEmptyBackgroundInjection(unittest.TestCase):
    """Check zero-background output for successful, absent, and failed injections."""

    def run_mixed_injections(self, in_place=True, read_only_variance=False, **kwargs):
        ic = mock.MagicMock()
        ic.__len__.return_value = 3
        ic.data = Table({"dataId": [0, 1, 2], "mjd_mid": [59000.0, 59001.0, 59002.0]})
        ic.get_standardizers.return_value = [{"std": SimpleNamespace()} for _ in range(3)]

        def make_exposure():
            exposure = SimpleNamespace(
                image=SimpleNamespace(array=np.full((2, 2), 10.0, dtype=np.float32)),
                variance=SimpleNamespace(array=np.full((2, 2), 100.0, dtype=np.float32)),
                mask=SimpleNamespace(array=np.ones((2, 2), dtype=np.int32)),
                psf=None,
                photoCalib=None,
                wcs=None,
            )
            if read_only_variance:
                exposure.variance.array.setflags(write=False)
            return exposure

        butler = mock.Mock()
        butler.get_dataset.side_effect = lambda did, **kwargs: SimpleNamespace(
            id=did, makeComponentRef=lambda name: (name, did)
        )

        def get_dataset(ref):
            if isinstance(ref, tuple):
                return SimpleNamespace(
                    date=SimpleNamespace(
                        toAstropy=lambda: Time(ic.data["mjd_mid"][ref[1]], format="mjd", scale="utc")
                    )
                )
            return make_exposure()

        butler.get.side_effect = get_dataset
        # The three exposures cover success, no catalog rows, and a rendering failure.
        catalog = Table({"injection_id": [0, 1], "obstime": [59000.0, 59002.0]})

        def inject(injection_catalogs, input_exposure, **kwargs):
            # Gain inference must still receive the real image and unscaled variance.
            np.testing.assert_array_equal(input_exposure.image.array, 10.0)
            np.testing.assert_array_equal(input_exposure.variance.array, 100.0)
            if injection_catalogs["obstime"][0] == 59002.0:
                raise RuntimeError("No sources were injected within bounds.")
            exposure = input_exposure if in_place else make_exposure()
            exposure.image.array[0, 0] += 2.0
            return SimpleNamespace(output_exposure=exposure, output_catalog=injection_catalogs)

        with (
            mock.patch("kbmod.injection.HAS_LSST", True),
            mock.patch("kbmod.injection.DatasetId", side_effect=int, create=True),
            mock.patch("kbmod.injection.VisitInjectConfig", MockVisitInjectConfig, create=True),
            mock.patch("kbmod.injection.VisitInjectTask", create=True) as task,
            mock.patch.object(
                ImageCollection,
                "fromStandardizers",
                side_effect=lambda stds: SimpleNamespace(data=ic.data.copy(), standardizers=stds),
            ),
        ):
            task.return_value.run.side_effect = inject
            with self.assertWarnsRegex(UserWarning, "had no objects successfully rendered"):
                rebuilt, injected_catalog = inject_sources_into_ic(ic, catalog, butler, **kwargs)
                standardizers = rebuilt.standardizers

        self.assertEqual(task.return_value.run.call_count, 2)
        self.assertEqual([std.ref.id for std in standardizers], [0, 1, 2])
        self.assertEqual(list(injected_catalog["injection_id"]), [0])
        # Every variance mode must preserve the original pixel masks.
        for std in standardizers:
            np.testing.assert_array_equal(std.exp.mask.array, 1)
        return [std.exp for std in standardizers]

    def test_no_render_exposure_has_empty_background(self):
        exposures = self.run_mixed_injections(zero_background=True)
        np.testing.assert_array_equal(exposures[0].image.array, [[2.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(exposures[1].image.array, 0.0)
        np.testing.assert_array_equal(exposures[2].image.array, 0.0)
        for exposure in exposures:
            np.testing.assert_array_equal(exposure.variance.array, 100.0)

    def test_variance_scale_applies_to_every_exposure(self):
        # Support injectors that mutate their input or return a separate exposure.
        for in_place in [True, False]:
            with self.subTest(in_place=in_place):
                exposures = self.run_mixed_injections(
                    variance_scale=1e-4, zero_background=True, in_place=in_place
                )
                for exposure in exposures:
                    np.testing.assert_allclose(exposure.variance.array, 0.01)

    def test_default_preserves_science_background(self):
        exposures = self.run_mixed_injections(read_only_variance=True)
        np.testing.assert_array_equal(exposures[0].image.array, [[12.0, 10.0], [10.0, 10.0]])
        for exposure in exposures[1:]:
            np.testing.assert_array_equal(exposure.image.array, 10.0)
        for exposure in exposures:
            np.testing.assert_array_equal(exposure.variance.array, 100.0)

    def test_unit_variance_scale_does_not_write_pixels(self):
        # Read-only arrays catch even an unnecessary in-place multiplication by 1.0.
        with self.assertLogs("kbmod.injection", level="INFO") as logs:
            exposures = self.run_mixed_injections(
                zero_background=True, variance_scale=1.0, read_only_variance=True
            )
        for exposure in exposures:
            np.testing.assert_array_equal(exposure.variance.array, 100.0)
            self.assertFalse(exposure.variance.array.flags.writeable)
        self.assertIn("zero_background=True", "\n".join(logs.output))
        self.assertIn("Skipping variance scaling (variance_scale=1.0)", "\n".join(logs.output))

    def test_logs_applied_variance_scale(self):
        with self.assertLogs("kbmod.injection", level="INFO") as logs:
            self.run_mixed_injections(zero_background=True, variance_scale=1e-4)
        self.assertIn("Applying variance_scale=0.0001 to all 3 returned exposures", "\n".join(logs.output))

    def test_variance_scale_preserves_science_background(self):
        exposures = self.run_mixed_injections(variance_scale=1e-4)
        np.testing.assert_array_equal(exposures[0].image.array, [[12.0, 10.0], [10.0, 10.0]])
        for exposure in exposures[1:]:
            np.testing.assert_array_equal(exposure.image.array, 10.0)
        for exposure in exposures:
            np.testing.assert_allclose(exposure.variance.array, 0.01)

    @mock.patch("kbmod.injection.HAS_LSST", True)
    def test_constant_variance_rejects_scaling(self):
        # None inputs ensure the conflict is rejected before accessing exposures.
        for scale in [1e-4, 2.0]:
            with self.subTest(variance_scale=scale):
                with self.assertRaisesRegex(ValueError, "constant_variance cannot be combined"):
                    inject_sources_into_ic(None, None, None, constant_variance=True, variance_scale=scale)

    def test_constant_variance_applies_after_injection_to_every_exposure(self):
        for zero_background in [False, True]:
            for in_place in [False, True]:
                with self.subTest(zero_background=zero_background, in_place=in_place):
                    with self.assertLogs("kbmod.injection", level="INFO") as logs:
                        exposures = self.run_mixed_injections(
                            constant_variance=True, zero_background=zero_background, in_place=in_place
                        )
                    for exposure in exposures:
                        np.testing.assert_array_equal(exposure.variance.array, 1.0)
                    # Constant variance must not change the chosen science-background behavior.
                    background = 0.0 if zero_background else 10.0
                    expected = np.full((2, 2), background)
                    expected[0, 0] += 2.0
                    np.testing.assert_array_equal(exposures[0].image.array, expected)
                    for exposure in exposures[1:]:
                        np.testing.assert_array_equal(exposure.image.array, background)
                    self.assertIn("Setting variance planes to constant 1.0", "\n".join(logs.output))

    @mock.patch("kbmod.injection.HAS_LSST", True)
    def test_invalid_variance_scale(self):
        for scale in [0.0, -1.0, np.nan, np.inf, -np.inf]:
            with self.subTest(variance_scale=scale):
                with self.assertRaisesRegex(ValueError, "variance_scale must be positive and finite"):
                    inject_sources_into_ic(None, None, None, variance_scale=scale)


class TestMatchInjectionResults(unittest.TestCase):
    def test_match_injection_results(self):
        """Test that match_injection_results correctly matches injected sources to trajectories."""

        # Create a simple WCS to convert between pixels and sky
        wcs_dict = {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1000.0,
            "CRPIX2": 1000.0,
            "CD1_1": -0.00001,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.00001,
        }
        wcs = WCS(wcs_dict)

        # Create a simulated KBMOD trajectory
        trj = Trajectory(x=50, y=50, vx=10, vy=0, flux=100.0)

        # Create a corresponding result object form this trajectory
        results = Results.from_trajectories([trj])
        results.wcs = wcs
        results.mjd_mid = np.array([59000.0, 59001.0, 59002.0])
        # match_injection_results -> KnownObjsMatcher requires the obs_valid array
        results.table["obs_valid"] = [np.array([True, True, True])]

        # Get skycoords from the trajectory at each obstime
        obstimes = [59000.0, 59001.0, 59002.0]
        base_t = obstimes[0]
        xs = [trj.x + trj.vx * (t - base_t) for t in obstimes]
        ys = [trj.y + trj.vy * (t - base_t) for t in obstimes]
        injected_coords = [wcs.pixel_to_world(x, y) for x, y in zip(xs, ys)]

        # shift our injected source by 0.1 arcseconds in RA and Dec
        injected_coords = [
            SkyCoord(ra=c.ra + 0.1 * u.arcsec, dec=c.dec + 0.1 * u.arcsec) for c in injected_coords
        ]

        # Create a catalog from the injected coordinates, with object name 101
        injected_obj_id = 101
        catalog = Table(
            {
                "obj_ids": [injected_obj_id, injected_obj_id, injected_obj_id],
                "obstime": obstimes,
                "ra": [c.ra.deg for c in injected_coords],
                "dec": [c.dec.deg for c in injected_coords],
                "guess_distance": [None, None, None],
            }
        )

        # Match within 1 arcsecond
        matched_results, recovered_ids, missed_ids = match_injection_results(
            catalog, results, guess_distance=None, sep_thresh=1.0, min_obs=3
        )

        # Check that injected object ID "101" was extracted
        self.assertIn(str(injected_obj_id), recovered_ids)
        self.assertNotIn(str(injected_obj_id), missed_ids)

        # Verify the Results table columns were appended correctly
        self.assertIn("recovered_injected_sources_min_obs_3", matched_results.table.colnames)
        self.assertEqual(
            list(matched_results.table["recovered_injected_sources_min_obs_3"][0]), [str(injected_obj_id)]
        )

        # Without the obs_ratio kwarg no obs_ratio column is added.
        self.assertFalse(
            any("obs_ratio" in name for name in matched_results.table.colnames),
        )

        # Rebuild the results with the last observation marked invalid, so only
        # 2 of the object's 3 catalog observations can match: an obs_ratio of 2/3.
        results = Results.from_trajectories([trj])
        results.wcs = wcs
        results.mjd_mid = np.array(obstimes)
        results.table["obs_valid"] = [np.array([True, True, False])]

        # obs_ratio is a minimum, so a threshold below 2/3 still recovers the object.
        matched_results, _, _ = match_injection_results(
            catalog, results, guess_distance=None, sep_thresh=1.0, min_obs=2, obs_ratio=0.5
        )
        self.assertEqual(
            list(matched_results.table["recovered_injected_sources_obs_ratio_0.5"][0]),
            [str(injected_obj_id)],
        )

        # A threshold above 2/3 does not.
        matched_results, _, _ = match_injection_results(
            catalog, results, guess_distance=None, sep_thresh=1.0, min_obs=2, obs_ratio=0.9
        )
        self.assertEqual(list(matched_results.table["recovered_injected_sources_obs_ratio_0.9"][0]), [])


if __name__ == "__main__":
    unittest.main()
