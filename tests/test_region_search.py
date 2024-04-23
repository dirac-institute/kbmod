import unittest

# A path for our mock repository
MOCK_REPO_PATH = "far/far/away"

from unittest import mock
from utils import (
    ConvexPolygon,
    DatasetRef,
    DatasetId,
    dafButler,
    DimensionRecord,
    LonLat,
    MockButler,
    Registry,
)

from astropy import units as u
from astropy.coordinates import SkyCoord

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


class TestRegionSearch(unittest.TestCase):
    """
    Test the region search functionality.
    """

    def setUp(self):
        self.registry = Registry()
        self.butler = MockButler(MOCK_REPO_PATH, registry=self.registry)

        # For the default collections and dataset types, we'll just use the first two of each
        self.default_collections = self.butler.registry.queryCollections()[:2]
        self.default_datasetTypes = [dt.name for dt in self.butler.registry.queryDatasetTypes()][:2]

        self.rs = region_search.RegionSearch(
            MOCK_REPO_PATH,
            self.default_collections,
            self.default_datasetTypes,
            butler=self.butler,
        )

    def test_init(self):
        """
        Test that the region search object can be initialized.
        """
        rs = region_search.RegionSearch(MOCK_REPO_PATH, [], [], butler=self.butler, fetch_data_on_start=False)
        self.assertTrue(rs is not None)
        self.assertEqual(0, len(rs.vdr_data))  # No data should be fetched

    def test_init_with_fetch(self):
        """
        Test that the region search object can fetch data on initializaiton
        """
        rs = region_search.RegionSearch(
            MOCK_REPO_PATH,
            self.default_collections,
            self.default_datasetTypes,
            butler=self.butler,
            fetch_data_on_start=True,
        )
        self.assertTrue(rs is not None)

        data = rs.fetch_vdr_data()
        self.assertGreater(len(data), 0)

        # Verify that the appropraiate columns have been fetched
        expected_columns = set(["data_id", "region", "detector", "uri", "center_coord"])
        # Compute the set of differing columns
        diff_columns = set(expected_columns).symmetric_difference(data.keys())
        self.assertEqual(len(diff_columns), 0)

    def test_chunked_data_ids(self):
        """
        Test the helper function for chunking data ids for parallel processing
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

        # Verify that the appropraiate columns have been fetched
        expected_columns = set(["data_id", "region", "detector", "uri", "center_coord"])
        # Compute the set of differing columns
        diff_columns = set(expected_columns).symmetric_difference(vdr_data.keys())
        self.assertEqual(len(diff_columns), 0)

    def test_get_instruments(self):
        """
        Test that the instruments are retrieved correctly.
        """
        data_ids = self.rs.fetch_vdr_data()["data_id"]
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
        data_ids = self.rs.fetch_vdr_data()["data_id"]
        # Get the URIs

        def func(repo_path):
            return MockButler(repo_path)

        parallel_rs = region_search.RegionSearch(
            MOCK_REPO_PATH,
            self.default_collections,
            self.default_datasetTypes,
            butler=self.butler,
            # TODO Turn on after fixing pickle issue for mocked objects
        )

        uris = parallel_rs.get_uris(data_ids)
        self.assertTrue(len(uris) > 0)

    def test_get_center_ra_dec(self):
        """
        Test that the center RA and Dec are retrieved correctly.
        """
        region = self.rs.fetch_vdr_data()["region"][0]

        # Get the center RA and Dec
        center_ra_dec = self.rs.get_center_ra_dec(region)
        self.assertTrue(len(center_ra_dec) > 0)

    def test_find_overlapping_coords(self):
        """
        Tests that we can find discrete piles with overlapping coordinates
        """
        # Create a set of regions that we can then greedily convert into discrete
        # piles within a radius threshold
        regions = []
        regions.append(
            ConvexPolygon(
                [
                    (-0.8572310214106003, 0.5140136573995331, 0.03073981031324692),
                    (-0.8573126243779648, 0.514061814910292, 0.027486624625501416),
                    (-0.8603167349539873, 0.5090182552169222, 0.027486931695458513),
                    (-0.8602353512965948, 0.508969729485055, 0.03074011788419143),
                ],
                center=LonLat(2.604388763115912, 0.029117535741884262),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8709063227199754, 0.49048670961887364, 0.030740278684830185),
                    (-0.8709892077685559, 0.49053262854455665, 0.0274870930414856),
                    (-0.8738549039182039, 0.48540915703478454, 0.027487400111442725),
                    (-0.8737722280742154, 0.48536286405417617, 0.030740586255774707),
                ],
                center=LonLat(2.6316151722984484, 0.02911800433559046),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8632807208053553, 0.5017969238521098, 0.054279317408622074),
                    (-0.8634293237866446, 0.5018822737832265, 0.051029266970199966),
                    (-0.8663644987606522, 0.49679828690228733, 0.05102957395625042),
                    (-0.8662161100467123, 0.49671256579311107, 0.05427962489522404),
                ],
                center=LonLat(2.6180008039930964, 0.05267892959436134),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8572306476090913, 0.5140142807953888, 0.03073981031329064),
                    (-0.8573122505414348, 0.5140624383654911, 0.027486624625545138),
                    (-0.8603163647852367, 0.5090188808567737, 0.027486931695502228),
                    (-0.8602349811631328, 0.5089703550657229, 0.030740117884235148),
                ],
                center=LonLat(2.604388035895388, 0.029117535741927998),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8709066920283894, 0.490486083334642, 0.030739808639786412),
                    (-0.8709895757751559, 0.4905320014532335, 0.027486622951882263),
                    (-0.8738552681990356, 0.4854085278595403, 0.027486930021839356),
                    (-0.8737725936565892, 0.4853622356858718, 0.030740116210730917),
                ],
                center=LonLat(2.631615899518966, 0.029117534067630124),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8632815518374847, 0.5017949720584495, 0.05428414385365578),
                    (-0.8634301686036089, 0.5018803295307347, 0.051034094243599164),
                    (-0.8663653328545027, 0.4967963364589571, 0.0510344012296149),
                    (-0.8662169303549427, 0.49671060780815945, 0.05428445134022294),
                ],
                center=LonLat(2.618002912932494, 0.052683763175059205),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8572305070104566, 0.514014197049692, 0.030745131028441175),
                    (-0.8573121248210437, 0.5140623634817817, 0.027491945845133575),
                    (-0.8603162390634397, 0.5090188059722266, 0.027492252915090963),
                    (-0.86023484056309, 0.5089702713191868, 0.03074543859938591),
                ],
                center=LonLat(2.604388035895447, 0.02912285898079983),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8709079891032256, 0.4904834774301412, 0.030744639926530638),
                    (-0.8709908867073091, 0.4905294029836303, 0.027491454696659565),
                    (-0.8738565642262993, 0.4854059210532718, 0.027491761766616923),
                    (-0.8737738758254457, 0.48535962144409595, 0.030744947497475358),
                ],
                center=LonLat(2.631618808401125, 0.0291223676459133),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.863282787712102, 0.5017930024533144, 0.05428269640419386),
                    (-0.8634314005967695, 0.5018783572241806, 0.05103264654570269),
                    (-0.8663665530170519, 0.4967943573222024, 0.051032953531728834),
                    (-0.8662181543980857, 0.4967086313723255, 0.05428300389077146),
                ],
                center=LonLat(2.618005240038212, 0.052682313585480325),
            )
        )

        regions.append(
            ConvexPolygon(
                [
                    (-0.8572327018287437, 0.5140106819036223, 0.03074270327029348),
                    (-0.8573143130502634, 0.5140588439538368, 0.027489517856807404),
                    (-0.8603184063869375, 0.5090152739921819, 0.027489824926764658),
                    (-0.8602370144741296, 0.5089667437201095, 0.030743010841238115),
                ],
                center=LonLat(2.604392181052405, 0.02912043007100976),
            )
        )

        # Take the above regions have and construct them as DimensionRecords within
        # our mock butler registry
        new_records = []
        for i, region in enumerate(regions):
            type = self.default_datasetTypes[i % 2]  # Use modulo 2 to alternate through the two dataset types
            new_records.append(DimensionRecord(f"dataId{i}", region, "fake_detector", type))
        self.registry.records = new_records

        # Fetch the VDR data for each of our 10 defined thresholds
        data = self.rs.fetch_vdr_data()
        self.assertEqual(len(data), 10)

        # Test that we can find 3 overlapping sets from the above test data
        radius_threshold = 30  # radius in arcseconds
        overlapping_sets = self.rs.find_overlapping_coords(data=data, uncertainty_radius=radius_threshold)
        self.assertEqual(len(overlapping_sets), 3)

        # Test that none of the indices are repeated across the sets
        prior_indices = set([])
        for s in overlapping_sets:
            for idx in s:
                self.assertNotIn(idx, prior_indices)
                prior_indices.add(idx)

        # Test that for each set, the distances between all elements are within
        # the uncertainty radius
        for s in overlapping_sets:
            # For this test data each set should have more than one element
            self.assertGreater(len(s), 1)

            # Fetch the center coordinate for the index we chose from the VDR data
            center_coords = [self.rs.vdr_data[idx]["center_coord"] for idx in s]

            # Convert the center coordinates for this pile to SkyCoord objects
            ra_decs = SkyCoord(
                ra=[c[0] * u.degree for c in center_coords],
                dec=[c[1] * u.degree for c in center_coords],
            )

            # Compute the separation between all pairs of coordinates
            for i in range(len(ra_decs)):
                distances = ra_decs[i].separation(ra_decs)
                for d in distances:
                    # Check that the separations is within the radius threshold
                    self.assertLessEqual(d.arcsec, radius_threshold)


if __name__ == "__main__":
    unittest.main()
