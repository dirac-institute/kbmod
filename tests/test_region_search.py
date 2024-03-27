import unittest

import numpy as np
from utils.utils_for_tests import get_absolute_data_path

# A path for our mock repository
MOCK_REPO_PATH = "far/far/away"

from unittest import mock
from utils import DatasetRef, DatasetId, dafButler, MockButler

# Mock out LSST so we can safely import kbmod
import sys

sys.modules["lsst"] = mock.MagicMock()
sys.modules["lsst.daf.butler"] = mock.MagicMock()

from kbmod import region_search


@mock.patch.dict(
    "sys.modules",
    {
        "lsst.daf.butler": dafButler,
        "lsst.daf.butler.core.DatasetRef": DatasetRef,
        "lsst.daf.butler.core.DatasetId": DatasetId,
    },
)
class TestRegionSearch(unittest.TestCase):
    """
    Test the region search functionality.
    """

    def setUp(self):
        self.butler = MockButler(MOCK_REPO_PATH)
        self.default_collections = self.butler.registry.queryCollections()[:2]
        self.default_datasetTypes = [dt.name for dt in self.butler.registry.queryDatasetTypes()][:2]

        self.rs = region_search.RegionSearch(
            MOCK_REPO_PATH, self.default_collections, self.default_datasetTypes, butler=self.butler
        )  # parallel is False by default

        self.parallel_rs = region_search.RegionSearch(
            MOCK_REPO_PATH,
            self.default_collections,
            self.default_datasetTypes,
            butler=self.butler,
            # parallel=True # TODO re-enable when there's a better way to instantiate a mock butler
        )

    def test_init(self):
        """
        Test that the region search object can be initialized.
        """
        # TODO add tests for the butler and parallel flags
        rs = region_search.RegionSearch(MOCK_REPO_PATH, [], [], butler=self.butler, parallel=False)
        self.assertTrue(rs is not None)

    def test_init_with_fetch(self):
        """
        Test that the region search object can fetch data on initializaiton
        """
        rs = region_search.RegionSearch(
            MOCK_REPO_PATH,
            self.default_collections,
            self.default_datasetTypes,
            butler=self.butler,
            fetch_data=True,
        )

        self.assertTrue(rs is not None)
        data = rs.fetch_vdr_data()
        self.assertGreater(len(data), 0)

    def test_chunked_data_ids(self):
        """
        Test that the data_ids are chunked correctly.
        """
        # Generate a list of random data_ids
        data_ids = [str(i) for i in range(100)]
        chunk_size = 10
        # Get all chunks from the generator
        chunks = [id for id in region_search._chunked_data_ids(data_ids, chunk_size)]

        for i in range(len(chunks)):
            chunk = chunks[i]
            self.assertEqual(len(chunk), chunk_size)
            for j in range(len(chunk)):
                self.assertEqual(chunk[j], data_ids[i * chunk_size + j])

    def test_get_collection_names(self):
        """
        Test that the collection names are retrieved correctly.
        """
        with self.assertRaises(ValueError):
            region_search.RegionSearch.get_collection_names(butler=None, repo_path=None)

        self.assertGreater(
            len(
                region_search.RegionSearch.get_collection_names(butler=self.butler, repo_path=MOCK_REPO_PATH)
            ),
            0,
        )

    def test_set_collections(self):
        """
        Test that the desired collections are set correctly.
        """
        collection_names = region_search.RegionSearch.get_collection_names(
            butler=self.butler, repo_path=MOCK_REPO_PATH
        )
        self.rs.set_collections(collection_names)
        self.assertEqual(self.rs.collections, collection_names)

    def test_get_dataset_type_freq(self):
        """
        Test that the dataset type frequency is retrieved correctly.
        """
        freq = self.rs.get_dataset_type_freq(butler=self.butler, collections=self.default_collections)
        self.assertTrue(len(freq) > 0)
        for dataset_type in freq:
            self.assertTrue(freq[dataset_type] > 0)

    def test_set_dataset_types(self):
        """
        Test that the desired dataset types are correctly set.
        """
        freq = self.rs.get_dataset_type_freq(butler=self.butler, collections=self.default_collections)

        self.assertGreater(len(freq), 0)
        dataset_types = list(freq.keys())[0]
        self.rs.set_dataset_types(dataset_types=dataset_types)

        self.assertEqual(self.rs.dataset_types, dataset_types)

    def test_fetch_vdr_data(self):
        """
        Test that the VDR data is retrieved correctly.
        """
        # Get the VDR data
        vdr_data = self.rs.fetch_vdr_data()
        self.assertTrue(len(vdr_data) > 0)

    def test_get_instruments(self):
        """
        Test that the instruments are retrieved correctly.
        """
        data_ids = self.rs.fetch_vdr_data()["data_id"].values
        # Get the instruments
        first_instrument = self.rs.get_instruments(data_ids, first_instrument_only=True)
        self.assertEqual(len(first_instrument), 1)

        # Now test the default where getting the first instrument is False.
        instruments = self.rs.get_instruments(data_ids)
        self.assertGreater(len(instruments), 1)

    def test_get_uris_serial(self):
        """
        Test that the URIs are retrieved correctly in serial mode.
        """
        data_ids = self.rs.fetch_vdr_data()["data_id"]
        # Get the URIs
        uris = self.rs.get_uris(data_ids)
        self.assertTrue(len(uris) > 0)

    def test_get_uris_parallel(self):
        """
        Test that the URIs are retrieved correctly in parallel mode.
        """
        data_ids = self.parallel_rs.fetch_vdr_data()["data_id"]
        # Get the URIs
        uris = self.parallel_rs.get_uris(data_ids)
        self.assertTrue(len(uris) > 0)

    def test_get_center_ra_dec(self):
        """
        Test that the center RA and Dec are retrieved correctly.
        """
        region = self.rs.fetch_vdr_data()["region"][0]

        # Get the center RA and Dec
        center_ra_dec = self.rs.get_center_ra_dec(region)
        self.assertTrue(len(center_ra_dec) > 0)
