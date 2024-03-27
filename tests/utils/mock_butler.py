import unittest
from unittest import mock

# TODO remove unneeded imports
import os
import uuid
import tempfile
import unittest
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
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def getCenter(self):
        return LonLat(self.x + self.width / 2, self.y + self.height / 2)


class ConvexPolygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def getBoundingBox(self):
        x = min([v[0] for v in self.vertices])
        y = min([v[1] for v in self.vertices])
        width = max([v[0] for v in self.vertices]) - x
        height = max([v[1] for v in self.vertices]) - y
        return Box(x, y, width, height)


class DimensionRecord:
    def __init__(self, dataId, region, detector):
        self.dataId = dataId
        self.region = region
        self.detector = detector


class Registry:
    def getDataset(self, ref):
        return ref

    def queryDimensionRecords(self, type, **kwargs):
        region1 = ConvexPolygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        region2 = ConvexPolygon([(1, 1), (1, 3), (3, 3), (3, 1)])
        return [
            DimensionRecord("dataId1", region1, "detector_replace_me"),
            DimensionRecord("dataId2", region2, "detector_replace_me"),
        ]

    # Fix queryCollections
    def queryCollections(self, **kwargs):
        return ["replace_me", "replace_me2"]

    def queryDatasetTypes(self, **kwargs):
        return [
            DatasetType("dataset_type_replace_me"),
            DatasetType("dataset_type_replace_me2"),
            DatasetType("dataset_type_replace_me3"),
        ]

    def queryDatasets(self, dataset_type, **kwargs):
        return DatasetQueryResults(
            [
                DatasetRef("dataset_ref_replace_me"),
                DatasetRef("dataset_ref_replace_me2"),
                DatasetRef("dataset_ref_replace_me3"),
            ]
        )


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

    def __init__(self, root, ref=None, mock_images_f=None):
        self.datastore = Datastore(root)
        self.registry = Registry()
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

    DatasetRef = DatasetRef
    DatasetId = DatasetId
    Butler = MockButler
