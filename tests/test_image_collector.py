import unittest

# A path for our mock repository
MOCK_REPO_PATH = "far/far/away"

from unittest import mock
from kbmod.image_collector import ImageCollector


class TestImageCollector(unittest.TestCase):
    """
    Test ingestion of butler collections into KBMOD ImageCollections.
    """

    def setUp(self):
        self.repo = MOCK_REPO_PATH
        self.image_collector = ImageCollector(self.repo)

    @mock.patch("lsst.daf.butler.Butler")
    def test_get_butler_collections(self, mock_butler):
        """
        Test that the collections are retrieved correctly.
        """
        mock_butler_instance = mock_butler.return_value
        mock_butler_instance.registry.queryCollections.return_value = [
            "collection1",
            "collection2",
            "weird_collection",
        ]
        # TODO change mocking such that each collection1 and collection 2 return different dataset types
        mock_butler_instance.registry.queryDatasets.return_value.count.return_value = 10

        # Test querying for collections with no matches
        collections = self.image_collector.get_collection_sizes("datasetType", collection_regex="bad.*")
        self.assertEqual(len(collections), 0)

        # Test querying for collections with matches
        collections = self.image_collector.get_collection_sizes(
            "datasetType", collection_regex="collection.*"
        )
        self.assertEqual(len(collections), 2)
        self.assertIn("collection1", collections)
        self.assertIn("collection2", collections)
        self.assertEqual(collections["collection1"], 10)
        self.assertEqual(collections["collection2"], 10)

        # Test querying for all collections
        collections = self.image_collector.get_collection_sizes("datasetType", collection_regex=".*")
        self.assertEqual(len(collections), 3)
        self.assertIn("collection1", collections)
        self.assertIn("collection2", collections)
        self.assertIn("weird_collection", collections)

        # Test specifying specific collection names
        collections = self.image_collector.get_collection_sizes(
            "datasetType", collection_names=["collection1"]
        )
        self.assertEqual(len(collections), 1)
        self.assertIn("collection1", collections)
        self.assertEqual(collections["collection1"], 10)

        collections = self.image_collector.get_collection_sizes(
            "datasetType", collection_names=["collection1", "collection2"]
        )
        self.assertEqual(len(collections), 2)
        self.assertIn("collection1", collections)
        self.assertIn("collection2", collections)
        self.assertEqual(collections["collection1"], 10)
        self.assertEqual(collections["collection2"], 10)

    @mock.patch("lsst.daf.butler.Butler")
    @mock.patch("kbmod.ImageCollection.fromTargets")
    def test_ingest_collections(self, mock_from_targets, mock_butler):
        """
        Test that collections are ingested correctly.
        """
        mock_butler.registry.queryDatasets.return_value = [mock.Mock()]

        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        self.image_collector.ingest_collections(
            collections,
            datasetType,
            output_dir=output_dir,
            overwrite=True,
        )
        self.assertEqual(mock_from_targets.call_count, 2)
