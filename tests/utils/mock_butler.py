from unittest import mock

import uuid

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
class Datastore:
    def __init__(self, root):
        self.root = root


class DatasetType:
    def __init__(self, name):
        self.name = name


class DatasetRef:
    def __init__(self, ref):
        self.ref = ref
        self.run = ref
        self.dataId = ref


class DatasetId:
    def __init__(self, ref):
        self.id = ref
        self.ref = ref
        self.run = ref


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
        self.collection = collection


class Registry:

    def __init__(self, records=None, **kwargs):
        if records is None:
            # Create some default records to return
            region1 = ConvexPolygon([(0, 0), (0, 1), (1, 1), (1, 0)], LonLat(0.5, 1))
            region2 = ConvexPolygon([(1, 1), (1, 3), (3, 3), (3, 1)], LonLat(0, 0.5))
            records = [
                DimensionRecord(DatasetRef("dataId1"), region1, "fake_detector", "type1", "collection1"),
                DimensionRecord(DatasetRef("dataId2"), region2, "fake_detector", "type2", "collection2"),
            ]
        self.records = records

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
    """

    def __init__(self, root, ref=None, mock_images_f=None, registry=None):
        self.datastore = Datastore(root)
        self.registry = Registry() if registry is None else registry
        self.mockImages = mock_images_f

    def getURI(self, ref, dataId=None, collections=None):
        mocked = mock.Mock(name="ButlerURI")
        mocked.geturl.return_value = f"file:/{self.datastore.root}"
        return mocked

    def getDataset(self, datid):
        return self.get(datid)

    def get(self, ref, collections=None, dataId=None):
        orig_ref = ref

        # Butler.get gets a DatasetRef, but can take an DatasetRef or DatasetId
        # DatasetId is type alias for UUID's, which are hex-strings when
        # serialized. We short it to an integer, because We use an integer to
        # read a particular file in FitsFactory. This means we got to cast
        # all these different objects to int (somehow). Firstly, it's one of
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

        # Finally we can proceed with mocking. Butler.get (the way we use it at
        # least) returns an Exposure[F/I/...] object. Exposure is like our
        # LayeredImage. We need to mock every attr, method and property that we
        # call the standardizer. We shortcut the results to match the KBMODV1.
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
