import os
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
                                 KBMODV1,
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
    empty_array = np.zeros((5, 5), dtype=np.float32)

    def __init__(self, root, ref=None, mock_images_f=None):
        self.datastore = Datastore(root)
        self.registry = Registry()
        self.current_ref = None
        self.mockImages = self.mockImagesDefault if mock_images_f is None else mock_images_f

    @classmethod
    def mockImagesDefault(self, mockedexp):
        mockedexp.image.array = self.empty_array
        mockedexp.variance.array = self.empty_array
        mockedexp.mask.array = self.empty_array.astype(np.int32)

    def getURI(self, ref, collections=None):
        mocked = mock.Mock(name="ButlerURI")
        mocked.geturl.return_value = f"file:/{self.datastore.root}"
        return mocked

    def getDataset(self, datid):
        return self.get(datid)

    def get(self, ref, collections=None):
        # Butler.get gets a DatasetRef in the actual code, we shorted that to
        # to be an integer, but it's in an attribute. We use the integer to
        # read a particular line from the serialized headers (same ones as
        # KBMODV1 tests). We want this to be tracked automatically if possible
        ref = ref if isinstance(ref, int) else ref.ref
        self.current_ref = ref

        hdul = FitsFactory.create_fits(ref % FitsFactory.n_files)
        prim = hdul["PRIMARY"].header

        # Butler get (the way we use it at least) is in the context of
        # returning an Exposure object. Exposure is like our LayeredImage. Now
        # we need to mock every attribute, method and property that we use in
        # the standardizer. We shortcut the results to match the KBMODV1.
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

        # WCS doesn't roundtrip NAXIS to/from headers and strings. Another
        # problem is that Rubin Sci. Pipes. return their own internal SkyWcs
        # object. Here we return a Header because that'll work with ButlerStd.
        # What happens if SkyWcs changes though? Relax to_header to stop the
        # warnings from non-standard SIP keywords
        wcshdr = WCS(hdul[1].header).to_header(relax=True)
        wcshdr["NAXIS1"] = hdul[1].header["NAXIS1"]
        wcshdr["NAXIS2"] = hdul[1].header["NAXIS2"]
        mocked.hasWcs.return_value = True
        mocked.wcs.getFitsMetadata.return_value = wcshdr

        # Mocking the images
        self.mockImages(mocked)

        # Same issue as with WCS, what if/when the mask changes
        # there's a change in definition of the mask plane compared to
        # DECam as only the exponent, not the full int representation
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
        self.img = MockButler.empty_array
        self.variance = MockButler.empty_array
        self.mask = MockButler.empty_array.astype(np.int32)

    def test_init(self):
        """Test ButlerStandardizer can be built from DatasetRef, DatasetId and
        the dataset id."""
        # Just makes sure no errors are raised, whether it actually does what
        # we want is tested later.
        _ = ButlerStandardizer(1, butler=self.butler)
        _ = ButlerStandardizer(DatasetRef(2), butler=self.butler)
        _ = ButlerStandardizer(DatasetId(3), butler=self.butler)

        _ = Standardizer.get(4, butler=self.butler)
        _ = Standardizer.get(DatasetRef(5), butler=self.butler)
        _ = Standardizer.get(DatasetId(6), butler=self.butler)

        _ = Standardizer.get(DatasetId(6), butler=self.butler,
                             force=ButlerStandardizer)

    def test_standardize(self):
        """Test ButlerStandardizer instantiates and standardizes as expected."""
        std = Standardizer.get(7, butler=self.butler)
        standardized = std.standardize()

        fits = FitsFactory.create_fits(7)
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
        np.testing.assert_equal([self.img, ], standardized["science"])
        np.testing.assert_equal([self.variance, ], standardized["variance"])
        np.testing.assert_equal([self.mask, ], standardized["mask"])

        # these are not easily comparable so just assert they exist
        self.assertTrue(standardized["meta"]["wcs"])
        self.assertTrue(standardized["meta"]["bbox"])

    def mock_kbmodv1like_bitmasking(self, mockedexp):
        """Assign each flag value to a pixel, standardize, then expect the mask
        only masked the values found in ButlerStdConfig.masked_flags expanded
        by grow_mask.

        Because Rubin keeps flag map in the FITS file headers the
        ButlerStdConfig does not contain them. We mock these to match the
        DECam KBMODV1-like flags in MockButler. So we set pixels to those
        flag values.
        """
        mockedexp.image.array = self.img
        mockedexp.variance.array = self.variance

        mask_arr = self.mask
        for i, flag in enumerate(KBMODV1Config.bit_flag_map):
            mask_arr.ravel()[i] = KBMODV1Config.bit_flag_map[flag]
        mockedexp.mask.array = mask_arr.astype(np.int32)

    # These tests are the same as KBMODV1 because the two hadn't diverged yet
    def test_bitmasking(self):
        """Test masking with direct config works as expected."""
        butler = MockButler("/far/far/away",
                            mock_images_f=self.mock_kbmodv1like_bitmasking)

        conf = StandardizerConfig(grow_mask=False)
        std = Standardizer.get(8, butler=butler, config=conf)
        standardizedMask = std.standardizeMaskImage()

        for mask in standardizedMask:
            for i, flag in enumerate(KBMODV1Config.bit_flag_map):
                with self.subTest("Failed to mask expected", flag=flag):
                    if flag in ButlerStandardizerConfig.mask_flags:
                        self.assertEqual(mask.ravel()[i], True)
                    else:
                        self.assertEqual(mask.ravel()[i], False)

    def mock_kbmodv1like_thresholding(self, mockedexp):
        """Set image pixel [1, 1] to 1 and [2, 2] to 3. Set variance and mask
        to empty. If brightness_threshold config is 2, and grow_mask is False,
        expect  [2, 2] to be masked.
        """
        img = self.img.copy()
        img[1, 1] = 1
        img[2, 2] = 3
        mockedexp.image.array = img
        mockedexp.variance.array = self.variance
        mockedexp.mask.array = self.mask

    def test_threshold_masking(self):
        """Test brightness threshold masking. Test config overrides."""
        butler = MockButler("/far/far/away",
                            mock_images_f=self.mock_kbmodv1like_thresholding)

        conf = StandardizerConfig({
            "grow_mask": False,
            "do_threshold": True,
            "brightness_threshold": 2,
        })
        std = Standardizer.get(9, butler=butler, config=conf)
        mask = std.standardizeMaskImage()[0]

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[2, 2])

    def mock_kbmodv1like_growmask(self, mockedexp):
        """Flag image pixel [2, 2] as BAD, and expect grow_mask to grow that
        mask to all neighboring pixels. Again, because flags are not available
        through the butler, but exposures only, we mocked them to be the same
        like DECam KBMODV1 flags.
        """
        mockedexp.image.array = self.img
        mockedexp.variance.array = self.variance

        mask = self.mask.copy()
        mask[2, 2] = KBMODV1Config.bit_flag_map["BAD"]
        mockedexp.mask.array = mask

    def test_grow_mask(self):
        """Test mask grows as expected."""
        butler = MockButler("/far/far/away",
                            mock_images_f=self.mock_kbmodv1like_growmask)

        conf = StandardizerConfig({
            "grow_mask": True,
            "grow_kernel_shape": (3, 3)
        })
        std = Standardizer.get(10, butler=butler, config=conf)
        mask = std.standardizeMaskImage()[0]

        self.assertTrue(mask[1:3, 1:3].all())
        self.assertFalse(mask[:, 0].all())
        self.assertFalse(mask[0, :].all())
        self.assertFalse(mask[-1, :].all())
        self.assertFalse(mask[:, -1].all())

    def test_psf(self):
        """Test PSFs are created as expected. Test instance config overrides."""
        std = Standardizer.get(11, butler=self.butler)

        psf = std.standardizePSF()[0]
        self.assertIsInstance(psf, PSF)
        self.assertEqual(psf.get_std(), std.config["psf_std"])



if __name__ == "__main__":
    unittest.main()
