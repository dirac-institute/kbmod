import os
import uuid
import tempfile
import unittest
from unittest import mock

from astropy.time import Time
from astropy.wcs import WCS
import numpy as np

from utils import DECamImdiffFactory
from kbmod import PSF, Standardizer, StandardizerConfig
from kbmod.standardizers import (ButlerStandardizer,
                                 ButlerStandardizerConfig,
                                 KBMODV1Config)


# Use a shared factory so that we can reference the same fits files in mocks
# and tests without having to untar the archive multiple times.
FitsFactory = DECamImdiffFactory()


# Patch Rubin Middleware out of existence
class Registry:
    def getDataset(self, ref):
        return ref


class Datastore:
    def __init__(self, root):
        self.root = root


class DatasetRef:
    def __init__(self, ref):
        self.ref = ref
        self.run = ref


class DatasetId:
    def __init__(self, ref):
        self.id = ref
        self.ref = ref
        self.run = ref


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
        self.current_ref = None
        self.mockImages = mock_images_f

    def getURI(self, ref, collections=None):
        mocked = mock.Mock(name="ButlerURI")
        mocked.geturl.return_value = f"file:/{self.datastore.root}"
        return mocked

    def getDataset(self, datid):
        return self.get(datid)

    def get(self, ref, collections=None):
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
                pass

        # Cast to int to cover for all eventualities
        ref = int(ref)
        self.current_ref = ref

        # Finally we can proceed with mocking. Butler.get (the way we use it at
        # least) returns an Exposure[F/I/...] object. Exposure is like our
        # LayeredImage. We need to mock every attr, method and property that we
        # call the standardizer. We shortcut the results to match the KBMODV1.
        hdul = FitsFactory.get_fits(ref % FitsFactory.n_files,
                                    spoof_data=True)
        prim = hdul["PRIMARY"].header

        mocked = mock.Mock(
            name="Exposure",
            spec_set=["visitInfo", "info", "hasWcs",
                      "getWidth", "getHeight", "getFilter",
                      "image", "variance", "mask", "wcs"]
        )

        # General metadata mocks
        mocked.visitInfo.date.toAstropy.return_value = \
            Time(hdul["PRIMARY"].header["DATE-AVG"], format="isot")
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


@mock.patch.dict("sys.modules", {
    "lsst.daf.butler.core.DatasetRef": DatasetRef,
    "lsst.daf.butler.core.DatasetId": DatasetId,
})
class TestButlerStandardizer(unittest.TestCase):
    """Test ButlerStandardizer. """

    def setUp(self):
        self.butler = MockButler("/far/far/away")

    def test_init(self):
        """Test ButlerStandardizer can be built from DatasetRef, DatasetId and
        the dataset id."""
        # Just makes sure no errors are raised, whether it actually does what
        # we want is tested later.
        _ = ButlerStandardizer(uuid.uuid1(), butler=self.butler)
        _ = ButlerStandardizer(uuid.uuid1().hex, butler=self.butler)
        _ = ButlerStandardizer(DatasetRef(2), butler=self.butler)
        _ = ButlerStandardizer(DatasetId(3), butler=self.butler)

        _ = Standardizer.get(DatasetRef(5), butler=self.butler)
        _ = Standardizer.get(DatasetId(6), butler=self.butler)

        _ = Standardizer.get(DatasetId(6), butler=self.butler,
                             force=ButlerStandardizer)

    def test_standardize(self):
        """Test ButlerStandardizer instantiates and standardizes as expected."""
        std = Standardizer.get(DatasetId(7), butler=self.butler)
        standardized = std.standardize()

        fits = FitsFactory.get_fits(7, spoof_data=True)
        hdr = fits["PRIMARY"].header
        expected = {
            "mjd": Time(hdr["DATE-AVG"], format="isot").mjd,
            "filter": hdr["FILTER"],
            "id": "7",
            "exp_id": hdr["EXPID"],
            "location": "file://far/far/away"
            }

        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                self.assertEqual(v, standardized["meta"][k])

        # The CRVAL1/2 are with respect to the origin (CRPIX), Our center_ra
        # definition uses the pixel in the center of the CCD. The permissible
        # deviation should be on the scale of half a CCD's footprint, unless
        # it's DECam then it could be as big as half an FOV of the focal plane
        self.assertAlmostEqual(standardized["meta"]["ra"][0], fits[1].header["CRVAL1"], 1)
        self.assertAlmostEqual(standardized["meta"]["dec"][0], fits[1].header["CRVAL2"], 1)

        # compare standardized images
        np.testing.assert_equal([fits["IMAGE"].data, ], standardized["science"])
        np.testing.assert_equal([fits["VARIANCE"].data, ], standardized["variance"])
        np.testing.assert_equal([fits["MASK"].data, ], standardized["mask"])

        # these are not easily comparable so just assert they exist
        self.assertTrue(standardized["meta"]["wcs"])
        self.assertTrue(standardized["meta"]["bbox"])

    def test_roundtrip(self):
        """Test ButlerStandardizer can instantiate itself from standardized
        data and a Data Butler."""
        std = Standardizer.get(DatasetId(8), butler=self.butler)
        standardized = std.standardize()

        std2 = ButlerStandardizer(**standardized["meta"], butler=self.butler)
        self.assertIsInstance(std, ButlerStandardizer)

    def mock_kbmodv1like_bitmasking(self, mockedexp):
        """Assign each flag that exists to a pixel, standardize, then expect
        the mask to only contain those pixels that are also in mask_flags.
        The grow_kernel is so large by default it would mask the nearly the
        whole image, so we turn it off.

        Because Rubin keeps flag map in the FITS file headers the
        ButlerStdConfig does not contain them. We mock these to match the
        DECam KBMODV1-like flags in MockButler, so we can set pixels to those
        flag values here.
        """
        mask_arr = mockedexp.mask.array
        for i, flag in enumerate(KBMODV1Config.bit_flag_map):
            mask_arr.ravel()[i] = KBMODV1Config.bit_flag_map[flag]

    # These tests are the same as KBMODV1 because the two hadn't diverged yet
    def test_bitmasking(self):
        """Test masking with direct config works as expected."""
        butler = MockButler("/far/far/away",
                            mock_images_f=self.mock_kbmodv1like_bitmasking)

        conf = StandardizerConfig(grow_mask=False)
        std = Standardizer.get(DatasetId(9), butler=butler, config=conf)
        standardizedMask = std.standardizeMaskImage()

        for mask in standardizedMask:
            for i, flag in enumerate(KBMODV1Config.bit_flag_map):
                with self.subTest("Failed to mask expected", flag=flag):
                    if flag in ButlerStandardizerConfig.mask_flags:
                        self.assertEqual(mask.ravel()[i], True)
                    else:
                        self.assertEqual(mask.ravel()[i], False)

    def mock_kbmodv1like_thresholding(self, mockedexp):
        """Set image pixel [1, 1] to 1 and [2, 2] to 3."""
        mockedexp.image.array[1, 1] = 1
        mockedexp.image.array[2, 2] = 3

    def test_threshold_masking(self):
        """Test brightness threshold masking. Test config overrides."""
        butler = MockButler("/far/far/away",
                            mock_images_f=self.mock_kbmodv1like_thresholding)

        conf = StandardizerConfig({
            "grow_mask": False,
            "do_threshold": True,
            "brightness_threshold": 2,
        })
        std = Standardizer.get(DatasetId(10), butler=butler, config=conf)
        mask = std.standardizeMaskImage()[0]

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[2, 2])

    def mock_kbmodv1like_growmask(self, mockedexp):
        """Flag image pixel [2, 2] as BAD, and expect grow_mask to grow that
        mask to all neighboring pixels. Again, because flags are not available
        through the butler, but exposures only, we mocked them to be the same
        like DECam KBMODV1 flags.
        """
        mockedexp.mask.array[2, 2] = KBMODV1Config.bit_flag_map["BAD"]

    def test_grow_mask(self):
        """Test mask grows as expected."""
        butler = MockButler("/far/far/away",
                            mock_images_f=self.mock_kbmodv1like_growmask)

        conf = StandardizerConfig({
            "grow_mask": True,
            "grow_kernel_shape": (3, 3)
        })
        std = Standardizer.get(DatasetId(11), butler=butler, config=conf)
        mask = std.standardizeMaskImage()[0]

        self.assertTrue(mask[1:3, 1:3].all())
        self.assertFalse(mask[:, 0].all())
        self.assertFalse(mask[0, :].all())
        self.assertFalse(mask[-1, :].all())
        self.assertFalse(mask[:, -1].all())

    def test_psf(self):
        """Test PSFs are created as expected. Test instance config overrides."""
        std = Standardizer.get(DatasetId(11), butler=self.butler)

        psf = std.standardizePSF()[0]
        self.assertIsInstance(psf, PSF)
        self.assertEqual(psf.get_std(), std.config["psf_std"])


if __name__ == "__main__":
    unittest.main()
