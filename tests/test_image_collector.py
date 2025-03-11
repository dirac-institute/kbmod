import unittest

# A path for our mock repository
MOCK_REPO_PATH = "far/far/away"

from unittest import mock
from utils import (
    ConvexPolygon,
    DatasetRef,
    DatasetId,
    dafButler,
    # DimensionRecord,
    LonLat,
    MockButler,
    Registry,
)

from astropy import units as u
from astropy.coordinates import SkyCoord

from kbmod.image_collector import ImageCollector

from concurrent.futures import ProcessPoolExecutor

with mock.patch.dict(
    "sys.modules",
    {
        "lsst": mock.MagicMock(),  # General mock for the LSST package import
        "lsst.daf.butler": dafButler,
        "lsst.daf.butler.core.DatasetRef": DatasetRef,
        "lsst.daf.butler.core.DatasetId": DatasetId,
    },
):
    from kbmod import region_search
    from kbmod.image_collector import ImageCollector


class MockProcessPoolExecutor:
    def __init__(self, max_workers=None):  # Added max_workers parameter
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def submit(self, fn, *args, **kwargs):
        future = mock.Mock()
        try:
            result = fn(*args, **kwargs)
            future.result.return_value = result
        except Exception as e:
            future.result.side_effect = e
        return future

    def shutdown(self, wait=True):
        pass


"""
Unit tests for the ImageCollector class
"""


class TestImageCollector(unittest.TestCase):
    """
    Test ingestion of butler collections into KBMOD ImageCollections.
    """

    def setUp(self):
        self.repo = MOCK_REPO_PATH
        self.image_collector = ImageCollector(self.repo)
        self.mock_butler = MockButler(MOCK_REPO_PATH)

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
        collections = self.image_collector.get_collection_sizes(
            "datasetType", collection_regex="bad.*"
        )
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
        collections = self.image_collector.get_collection_sizes(
            "datasetType", collection_regex=".*"
        )
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

    # @mock.patch("kbmod.image_collector.dafButler.Butler")
    @mock.patch(
        "kbmod.image_collector.dafButler.Butler", new_callable=lambda: MockButler
    )
    @mock.patch("kbmod.ImageCollection.fromTargets")
    def test_ingest_collections_serial(self, mock_from_targets, mock_butler):
        """
        Test that collections are ingested correctly in serial mode.
        """
        # mock_butler_instance = mock_butler.return_value
        mock_butler.registry.queryDatasets.return_value = [mock.Mock()]

        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        self.image_collector.ingest_collections(
            collections,
            datasetType,
            is_parallel=False,
            output_dir=output_dir,
            overwrite=True,
        )
        self.assertEqual(mock_from_targets.call_count, 2)

    # @mock.patch("lsst.daf.butler.Butler")
    # @mock.patch("kbmod.ImageCollection.fromTargets")
    @mock.patch("concurrent.futures.ProcessPoolExecutor", wraps=MockProcessPoolExecutor)
    def test_sanity(self, mock_executor):  # , mock_from_targets):#, mock_butler):
        pass

    """
    @mock.patch("kbmod.image_collector.dafButler.Butler")
    def test_ingest_collections_parallel_stupid(self, mock_butler):
        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        mock_butler_instance = mock_butler.return_value
        mock_butler_instance.registry.queryDatasets.return_value = collections

        ics = self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, output_dir=output_dir, overwrite=True)
        self.assertEqual(len(ics), 2)
    """

    # @mock.patch("kbmod.image_collector.dafButler.Butler")
    @mock.patch(
        "kbmod.image_collector.dafButler.Butler", new_callable=lambda: MockButler
    )
    def test_ingest_collections_parallel(self, mock_butler):
        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        mock_butler.registry.queryDatasets.return_value = collections

        ics = self.image_collector.ingest_collections(
            collections,
            datasetType,
            is_parallel=True,
            output_dir=output_dir,
            overwrite=True,
        )
        self.assertEqual(len(ics), 2)

    """
    @mock.patch("kbmod.image_collector._ingest_collection", return_value="mock_image_collection")
    @mock.patch("concurrent.futures.ProcessPoolExecutor", wraps=MockProcessPoolExecutor)
    def test_ingest_collections_parallel(self, mock_executor, mock_ingest):
        #Test that collections are ingested correctly in parallel mode.
        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"
        naming_scheme = lambda name: f"custom_{name}.collection"

        results = self.image_collector.ingest_collections(
            collections,
            datasetType,
            is_parallel=True,
            output_dir=output_dir,
            naming_scheme=naming_scheme,
            overwrite=True
        )

        # Check that the _ingest_collection function was called with the correct arguments
        mock_ingest.assert_any_call((self.repo, "collection1", datasetType, output_dir, naming_scheme, True))
        mock_ingest.assert_any_call((self.repo, "collection2", datasetType, output_dir, naming_scheme, True))

        # Check the results
        self.assertEqual(results, ["mock_image_collection", "mock_image_collection"])

        # Check that the ProcessPoolExecutor was used
        self.asser
    """
    """
    @mock.patch("lsst.daf.butler.Butler")
    @mock.patch("kbmod.ImageCollection.fromTargets")
    @mock.patch("concurrent.futures.ProcessPoolExecutor", wraps=MockProcessPoolExecutor)
    def test_ingest_collections_parallel(self, mock_executor, mock_from_targets, mock_butler):
        # Test that collections are ingested correctly in parallel mode.
        def mock_ingest_collection(collection, datasetType, output_dir, overwrite):
            return ImageCollection.fromTargets([]) # TODO add some data

        # Setup butler mock
        mock_butler_instance = mock_butler.return_value
        mock_butler_instance.registry.queryDatasets.return_value = [mock.Mock()]

        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        # Create a spy to track calls to our mock function
        with mock.patch("kbmod.image_collector._ingest_collection", side_effect=mock_ingest_collection) as mock_ingest:
            with mock.patch("os.cpu_count", return_value=2):
                self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, 
                                                    output_dir=output_dir, overwrite=True)
                
                # Verify the mock was called twice (once for each collection)
                self.assertEqual(mock_ingest.call_count, 2)
                
                # Verify the exact arguments for each call
                expected_calls = [
                    mock.call(collection, datasetType, output_dir, True)
                    for collection in collections
                ]
                mock_ingest.assert_has_calls(expected_calls, any_order=True)  # Note: added any_order=True since parallel execution
    """
    """
    @mock.patch("kbmod.image_collector.dafButler.Butler")
    @mock.patch("kbmod.ImageCollection.fromTargets")
    @mock.patch("concurrent.futures.ProcessPoolExecutor")
    def test_ingest_collections_parallel(self, mock_executor, mock_from_targets, mock_butler):
        
        Test that collections are ingested correctly in parallel mode.
        
        mock_butler_instance = mock_butler.return_value
        mock_butler_instance.registry.queryDatasets.return_value = [mock.Mock()]

        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        # Mock the executor to return a mock future
        mock_future = mock.Mock()
        mock_future.result.return_value = None
        mock_executor_instance = mock_executor.return_value
        mock_executor_instance.submit.return_value = mock_future

        # Mock out os.cpu_count to return 2
        with mock.patch("os.cpu_count", return_value=2):
            self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, output_dir=output_dir, overwrite=True)
            self.assertEqual(mock_from_targets.call_count, 2)
            self.assertEqual(mock_executor_instance.submit.call_count, 2)

            # Now test when the number of collections is less than the number of workers
            collections = ["collection1"]
            self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, output_dir=output_dir, overwrite=True)
            self.assertEqual(mock_from_targets.call_count, 3)
            self.assertEqual(mock_executor_instance.submit.call_count, 3)

        # Test setting the number of workers to 2 by argument regardless of cpu count
        with mock.patch("os.cpu_count", return_value=4):
            self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, output_dir=output_dir, overwrite=True, num_workers=2)
            self.assertEqual(mock_from_targets.call_count, 4)
            self.assertEqual(mock_executor_instance.submit.call_count, 4)

        # Test handling of an empty list of collections
        self.image_collector.ingest_collections([], datasetType, is_parallel=True, output_dir=output_dir, overwrite=True)
        self.assertEqual(mock_from_targets.call_count, 4)
        self.assertEqual(mock_executor_instance.submit.call_count, 4)

        # Test handling of an invalid/missing collection
        collections = ["bad_collection"]
        self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, output_dir=output_dir, overwrite=True)
        self.assertEqual(mock_from_targets.call_count, 4)
        self.assertEqual(mock_executor_instance.submit.call_count, 4)

        # Test handling of an invalid/missing dataset type
        collections = ["collection1"]
        datasetType = "bad_datasetType"
        self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, output_dir=output_dir, overwrite=True)
        self.assertEqual(mock_from_targets.call_count, 4)
        self.assertEqual(mock_executor_instance.submit.call_count, 4)

    """

    """
    @mock.patch("kbmod.region_search.dafButler.Butler")
    @mock.patch("kbmod.ImageCollection.fromTargets")
    @mock.patch("concurrent.futures.ProcessPoolExecutor", wraps=MockThreadPoolExecutor)
    @mock.patch("kbmod.region_search._ingest_collection")  # Add this mock
    def test_ingest_collections_parallel(self, mock_ingest, mock_executor, mock_from_targets, mock_butler):
        #Test that collections are ingested correctly in parallel mode.
        # Setup butler mock
        mock_butler_instance = mock_butler.return_value
        mock_butler_instance.registry.queryDatasets.return_value = [mock.Mock()]

        # Setup executor mock

        mock_future = mock.Mock()
        mock_future.result.return_value = mock_ingest.return_value  # Use mock_ingest instead
        mock_executor_instance = mock_executor.return_value
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor_instance.submit.return_value = mock_future

        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"


        def mock_submit(*args, **kwargs):
            print("Submit called with args:", args)
            print("Submit called with kwargs:", kwargs)
            raise ValueError("Submit called")
            return mock_future
        mock_executor_instance.submit.side_effect = mock_submit
        

        # Mock out os.cpu_count to return 2
        with mock.patch("os.cpu_count", return_value=2):
            self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, 
                                                output_dir=output_dir, overwrite=True)
            #self.assertEqual(mock_executor_instance.submit.call_count, 2)
    """

    """
    def mock_ingest_collection(collection, datasetType, output_dir, overwrite):
        return f"Processed {collection}"

    @mock.patch("kbmod.region_search.dafButler.Butler")
    @mock.patch("kbmod.ImageCollection.fromTargets")
    @mock.patch("concurrent.futures.ProcessPoolExecutor", wraps=MockProcessPoolExecutor)
    @mock.patch("kbmod.region_search._ingest_collection", side_effect=mock_ingest_collection)
    def test_ingest_collections_parallel(self, mock_ingest, mock_executor, mock_from_targets, mock_butler):
        #Test that collections are ingested correctly in parallel mode.
        # Setup butler mock
        mock_butler_instance = mock_butler.return_value
        mock_butler_instance.registry.queryDatasets.return_value = [mock.Mock()]

        collections = ["collection1", "collection2"]
        datasetType = "datasetType"
        output_dir = "/tmp"

        with mock.patch("os.cpu_count", return_value=2):
            self.image_collector.ingest_collections(collections, datasetType, is_parallel=True, 
                                                output_dir=output_dir, overwrite=True)
            
            # Verify the mock was called twice (once for each collection)
            self.assertEqual(mock_ingest.call_count, 2)
            
            # Verify the exact arguments for each call
            expected_calls = [
                mock.call(collection, datasetType, output_dir, True)
                for collection in collections
            ]
            mock_ingest.assert_has_calls(expected_calls)

            # If you want to verify calls were made in a specific order:
            self.assertEqual(mock_ingest.mock_calls, expected_calls)
            
            # Or if you want to check individual call arguments:
            first_call_args = mock_ingest.call_args_list[0]
            self.assertEqual(first_call_args[0][0], "collection1")  # First positional arg
            self.assertEqual(first_call_args[0][1], datasetType)   # Second positional arg
    """
