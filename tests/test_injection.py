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
            "x",
            "y",
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


if __name__ == "__main__":
    unittest.main()
