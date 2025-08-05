import uuid
import copy

from unittest import mock

from kbmod.standardizers import KBMODV1Config

from astropy.time import Time
from astropy.wcs import WCS
import numpy as np

from .mock_fits import DECamImdiffFactory

__all__ = [
    "MockButler",
    "Registry",
    "Datastore",
    "DatasetRef",
    "DatasetId",
    "dafButler",
    "DimensionRecord",
    "ConvexPolygon",
    "LonLat",
]


# Patch Rubin Middleware out of existence
@mock.patch("__main__.uuid.UUID", spec=uuid.UUID)
class UUID:
    """Patch out actual UUID-s so that they become
    simple integers.

    Ultimately fake data will have to return an
    image and metadata. This is then practical way
    of keeping tract which mocked data structure
    belongs to which data, while also passing
    is-instance checks in standardizers.
    """

    def __init__(self, dataId):
        self.id = dataId
        self.ref = dataId
        self.hex = dataId

    def __str__(self):
        return str(self.id)


class Datastore:
    def __init__(self, root):
        self.root = root
        self.roots = [root]


class DatasetType:
    def __init__(self, name):
        self.name = name


class DatasetId:
    """In the stack this is a dict-like
    mapping of data dimensions of the dataset.

    For us, this is basically an integer. When correctly interpreting and
    translating metadata needs validating, you can fill in real-like deep data
    by setting ``fill_metadata`` to True, otherwise the string:
        'test_<metadata>'
    will be returned (f.e. dataId[visit] --> test_visit)
    """

    def __init__(self, ref, fill_metadata=False):
        self.id = ref
        self.ref = ref
        self.run = ref
        if fill_metadata:
            self.fill_metadata()

    def fill_metadata(self):
        # DataIds in the stack carry dimension information, this is band,
        # detector, visit etc. Some of these, f.e. time, bbox etc. we also
        # want to validate, so we need to fake this data anyhow. The values
        # don't really matter, as long as they have a predictable result after
        # being transformed and cover different test cases, so just associate
        # with closest matching header values.
        hdul = FitsFactory.get_fits(self.ref % FitsFactory.n_files)
        prim = hdul["PRIMARY"].header
        self.physical_filter = prim["FILTER"]
        self.band = self.physical_filter.split(" ")[0]
        self.visit = prim["EXPID"]
        self.detector = prim["CCDNUM"]

    def __getitem__(self, key):
        test = getattr(self, key, None)
        return f"test_{key}" if test is None else test

    def __str__(self):
        return str(self.id)


class DatasetRef:
    """Key object that can be used to fetch the
    dataset in the Rubin stack.

    The `ref` attribute keeps track of the data
    index we are at directly and that should match
    the the `dataId` or `id` attr values (but not
    types). Updating the ref in place requires you
    to sync those values.
    """

    def __init__(self, dataId):
        self.id = UUID(dataId)
        self.datasetType = DatasetType("test_datasettype_name")
        self.run = dataId.run
        self.dataId = dataId
        self.ref = dataId.ref

    def makeComponentRef(self, name):
        newref = copy.deepcopy(self)
        newref.datasetType.name += f".{name}"
        return newref


class PropertyList:
    def __init__(self, valdict):
        self.valdict = valdict

    def __getitem__(self, key):
        return self.valdict[key]

    def __contains__(self, key):
        return key in self.valdict

    def __delitem__(self, key):
        del self.valdict[key]


class DatasetQueryResults:
    def __init__(self, dataset_refs):
        self.refs = dataset_refs

    def count(self, **kwargs):
        return len(self.refs)


class Angle:
    def __init__(self, value):
        self.value = value

    def asDegrees(self):
        return self.value


class LonLat:
    def __init__(self, lon, lat):
        self.lon = Angle(lon)
        self.lat = Angle(lat)

    def getLon(self):
        return self.lon

    def getLat(self):
        return self.lat


class Box:
    def __init__(self, center):
        self.center = center

    def getCenter(self):
        return self.center


class ConvexPolygon:
    def __init__(self, vertices, center=None):
        self.vertices = vertices
        self.center = center

    def getBoundingBox(self):
        return Box(self.center)


class DimensionRecord:
    def __init__(
        self, dataId, region, detector, dataset_type="default_type", collection="default_collection"
    ):
        self.dataId = dataId
        self.region = region
        self.detector = detector
        self.dataset_type = DatasetType(dataset_type)
        self.collection = str(collection)


class Registry:
    def __init__(self, records=None, **kwargs):
        if records is None:
            # Create some default records to return
            region1 = ConvexPolygon([(0, 0), (0, 1), (1, 1), (1, 0)], LonLat(0.5, 1))
            region2 = ConvexPolygon([(1, 1), (1, 3), (3, 3), (3, 1)], LonLat(0, 0.5))
            records = [
                DimensionRecord(DatasetRef(DatasetId(1)), region1, "fake_detector", "type1", "collection1"),
                DimensionRecord(DatasetRef(DatasetId(2)), region2, "fake_detector", "type2", "collection2"),
            ]
        self.records = records

    def getDataset(self, dataId):
        return DatasetRef(dataId)

    def queryDimensionRecords(self, type, datasets=None, **kwargs):
        """Query the registry for records of a particular type 'datasets'. Optionally"""
        if datasets is None:
            return self.records
        if isinstance(datasets, DatasetType):
            datasets = datasets.name
        return [record for record in self.records if record.dataset_type.name == datasets]

    def queryCollections(self, **kwargs):
        """Query the registry for all collections."""
        return [record.collection for record in self.records]

    def queryDatasetTypes(self, **kwargs):
        """Query the registry for all dataset types."""
        return [record.dataset_type for record in self.records]

    def queryDatasets(self, dataset_type, **kwargs):
        """Query the registry for all datasets of a particular type."""
        if isinstance(dataset_type, DatasetType):
            dataset_type = dataset_type.name
        return DatasetQueryResults([r.dataId for r in self.records if r.dataset_type.name == dataset_type])


FitsFactory = DECamImdiffFactory()


class MockButler:
    """Mocked Vera C. Rubin Data Butler functionality sufficient to be used in
    a ButlerStandardizer.

    The mocked .get method will return an mocked Exposure object with all the,
    generally, expected attributes (info, visitInfo, image, variance, mask,
    wcs). Most of these attributes are mocked such that they return an integer
    id, which is then used in a FitsFactory to read out the serialized header
    of some underlying real data. Particularly, we target DECam, such that
    outputs of ButlerStandardizer and KBMODV1 are comparable.

    By default the mocked image arrays will contain the empty
    `Butler.empty_arrat` but providing a callable `mock_images_f`, that takes
    in a single mocked Exposure object, and assigns the:
    * mocked.image.array
    * mocked.variance.array
    * mocked.mask.array
    attributes can be used to customize the returned arrays.

    The mocked metadata will be a copy of the header of the first file in the
    FitsFactory, but with the option to remove some header keys by providing a
    list of keys in `missing_headers`.
    """

    def __init__(self, root, ref=None, mock_images_f=None, registry=None, missing_headers=[]):
        self.datastore = Datastore(root)
        self._datastore = Datastore(root)
        self.registry = Registry() if registry is None else registry
        self.mockImages = mock_images_f
        self.missing_headers = missing_headers

    def getURI(self, ref, dataId=None, collections=None):
        mocked = mock.Mock(name="ButlerURI")
        mocked.geturl.return_value = f"file:/{self.datastore.root}"
        return mocked

    def getDataset(self, datid):
        return self.get(datid)

    def get_dataset(self, datid, dimension_records=False, datastore_records=False):
        return DatasetRef(datid)

    def get(self, ref, collections=None, dataId=None):
        orig_ref = ref

        # this covers tests in region_search because they pass dataId (that are
        # actuall dataRefs) with a component name as ref. TODO: fix this?
        if dataId is not None and isinstance(ref, str):
            orig_ref = dataId.makeComponentRef(ref.split(".")[-1])

        # Butler.get gets a DatasetRef, but can take an DatasetRef or DatasetId
        # DatasetId is type alias for UUID's, which are hex-strings when
        # serialized. We short it to an integer because we use an integer to
        # read a particular file in FitsFactory. This means we somehow have to
        # cast all these different objects to int. Firstly, if it's one of
        # our mocks, dig out the value we really care about:
        if isinstance(ref, (DatasetId, DatasetRef)):
            ref = ref.ref

        # that value can be an int, a simple str(int) (used in testing only),
        # a large hex UUID string, or a UUID object. Duck-type them to int
        if isinstance(ref, uuid.UUID):
            ref = ref.int
        elif isinstance(ref, str):
            try:
                ref = uuid.UUID(ref).int
            except (ValueError, AttributeError):
                # likely a str(int)
                try:
                    ref = int(ref)
                except ValueError:
                    ref = len(ref)

        # then figure out what dataset was being mocked and build it
        if ".metadata" in orig_ref.datasetType.name:
            return self.mock_metadata(ref)
        elif ".visitInfo" in orig_ref.datasetType.name:
            return self.mock_visitinfo(ref)
        elif ".summaryStats" in orig_ref.datasetType.name:
            return self.mock_summarystats(ref)
        elif ".wcs" in orig_ref.datasetType.name:
            return self.mock_wcs(ref)
        elif ".bbox" in orig_ref.datasetType.name:
            return self.mock_bbox(ref)
        else:
            return self.mock_exposure(ref)

    def mock_metadata(self, ref):
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files)
        prim = hdul["PRIMARY"].header
        meta = PropertyList(dict(prim))
        if self.missing_headers:
            # remove some keys from the metadata to simulate different environments
            for key in self.missing_headers:
                del meta[key]
        return meta

    def mock_visitinfo(self, ref):
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files)
        prim = hdul["PRIMARY"].header

        mocked_visit = mock.Mock(name="VisitInfo")
        mocked_visit.exposureTime = prim["EXPREQ"]
        expstart = Time(prim["DATE-AVG"], format="isot", scale="tai")
        mocked_visit.date.toAstropy.return_value = expstart
        mocked_visit.date.toAstropy.return_value = expstart
        mocked_visit.date.return_value = expstart

        mocked_obs = mock.Mock(name="Observatory")
        mocked_obs.getLongitude.return_value = Angle(prim["OBS-LONG"])
        mocked_obs.getLatitude.return_value = Angle(prim["OBS-LAT"])
        mocked_obs.getElevation.return_value = prim["OBS-ELEV"]

        mocked_visit.getObservatory.return_value = mocked_obs

        return mocked_visit

    def mock_summarystats(self, ref):
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files)
        wcs = WCS(hdul[1].header)
        naxis1, naxis2 = hdul[1].header["NAXIS1"], hdul[1].header["NAXIS2"]

        mocked = mock.Mock(name="SummaryStats")
        mocked.psfSigma = 1.0
        mocked.psfArea = 1.0
        mocked.nPsfStar = 1.0
        mocked.skyBg = 1.0
        mocked.skyNoise = 1.0
        mocked.zeroPoint = 1.0
        mocked.astromOffsetMean = 1.0
        mocked.astromOffsetStd = 1.0

        mocked.effTime = 0
        mocked.effTimePsfSigmaScale = 0
        mocked.effTimeSkyBgScale = 0
        mocked.effTimeZeroPointScale = 0

        corners = [
            wcs.pixel_to_world(0, 0),
            wcs.pixel_to_world(naxis1, 0),
            wcs.pixel_to_world(naxis1, naxis2),
            wcs.pixel_to_world(0, naxis2),
        ]
        mocked.raCorners = [c.ra.deg for c in corners]
        mocked.decCorners = [c.dec.deg for c in corners]
        center = wcs.pixel_to_world(naxis1 // 2, naxis2 // 2)
        mocked.ra = center.ra.deg
        mocked.dec = center.dec.deg

        return mocked

    def mock_wcs(self, ref):
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files)
        mocked = mock.Mock(name="SkyWcs")

        mocked_coord = mock.Mock(name="RubinCoord")
        wcs = WCS(hdul[1].header)

        def fake_skywcs_transform(*args, **kwargs):
            coord = wcs.pixel_to_world(*args, **kwargs)
            mocked_angle = mock.Mock(name="RubinAngle")
            mocked_angle.asDegrees.return_value = coord.ra.deg
            mocked_coord.getRa.return_value = mocked_angle
            mocked_angle = mock.Mock(name="RubinAngle")
            mocked_angle.asDegrees.return_value = coord.dec.deg
            mocked_coord.getDec.return_value = mocked_angle
            return mocked_coord

        mocked.pixelToSky.side_effect = fake_skywcs_transform

        mocked.getFitsMetadata.return_value = hdul[1].header
        return mocked

    def mock_bbox(self, ref):
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files)
        mocked = mock.Mock(name="BBox")
        mocked.getWidth.return_value = hdul[1].header["NAXIS1"]
        mocked.getHeight.return_value = hdul[1].header["NAXIS2"]
        return mocked

    def mock_exposure(self, ref):
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files, spoof_data=True)
        prim = hdul["PRIMARY"].header

        mocked = mock.Mock(
            name="Exposure",
            spec_set=[
                "visitInfo",
                "info",
                "hasWcs",
                "getWidth",
                "getHeight",
                "getFilter",
                "image",
                "variance",
                "mask",
                "wcs",
            ],
        )

        # General metadata mocks
        mocked.visitInfo.date.toAstropy.return_value = Time(hdul["PRIMARY"].header["DATE-AVG"], format="isot")
        mocked.visitInfo.date.return_value = Time(hdul["PRIMARY"].header["DATE-AVG"], format="isot")
        mocked.info.id = prim["EXPID"]
        mocked.getWidth.return_value = hdul[1].header["NAXIS1"]
        mocked.getHeight.return_value = hdul[1].header["NAXIS2"]
        mocked.info.getFilter().physicalLabel = prim["FILTER"]

        # Rubin Sci. Pipes. return their own internal SkyWcs object. We mock a
        # Header that'll work with ButlerStd instead. It works because in the
        # STD we cast SkyWcs to dict-like thing, from which we make a WCS. What
        # happens if SkyWcs changes though?
        wcshdr = WCS(hdul[1].header).to_header(relax=True)
        wcshdr["NAXIS1"] = hdul[1].header["NAXIS1"]
        wcshdr["NAXIS2"] = hdul[1].header["NAXIS2"]
        mocked.hasWcs.return_value = True
        mocked.wcs.getFitsMetadata.return_value = wcshdr

        # Mocking the images consists of using the Factory default, then
        # invoking any user specified method on the mocked exposure obj.
        mocked.image.array = hdul["IMAGE"].data
        mocked.variance.array = hdul["VARIANCE"].data
        mocked.mask.array = hdul["MASK"].data
        if self.mockImages is not None:
            self.mockImages(mocked)

        # Same issue as with WCS, what if there's a change in definition of the
        # mask plane? Note the change in definition of a flag to exponent only.
        bit_flag_map = {}
        for key, val in KBMODV1Config.bit_flag_map.items():
            bit_flag_map[key] = int(np.log2(val))
        mocked.mask.getMaskPlaneDict.return_value = bit_flag_map

        return mocked


class dafButler:
    """Intercepts calls ``import lsst.daf.butler as dafButler`` and shortcuts
    them to our mocks.
    """

    DatasetType = DatasetType
    DatasetRef = DatasetRef
    DatasetId = DatasetId
    Butler = MockButler
