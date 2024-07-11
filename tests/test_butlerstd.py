import os
import uuid
import tempfile
import unittest
from unittest import mock

from astropy.time import Time
from astropy.wcs import WCS
import numpy as np

from utils import DECamImdiffFactory, MockButler, Registry, Datastore, DatasetRef, DatasetId, dafButler
from kbmod import PSF, Standardizer, StandardizerConfig
from kbmod.standardizers import ButlerStandardizer, ButlerStandardizerConfig, KBMODV1Config


# Use a shared factory so that we can reference the same fits files in mocks
# and tests without having to untar the archive multiple times.
FitsFactory = DECamImdiffFactory()


@mock.patch.dict(
    "sys.modules",
    {
        "lsst.daf.butler": dafButler,
        "lsst.daf.butler.core.DatasetRef": DatasetRef,
        "lsst.daf.butler.core.DatasetId": DatasetId,
    },
)
class TestButlerStandardizer(unittest.TestCase):
    """Test ButlerStandardizer."""

    def setUp(self):
        self.butler = MockButler("/far/far/away")

    def test_init(self):
        """Test ButlerStandardizer can be built from DatasetRef, DatasetId and
        the dataset id."""
        # Just makes sure no errors are raised, whether it actually does what
        # we want is tested later.
        _ = ButlerStandardizer(uuid.uuid1(), butler=self.butler)
        _ = ButlerStandardizer(uuid.uuid1().hex, butler=self.butler)
        _ = ButlerStandardizer(DatasetRef(DatasetId(2)), butler=self.butler)
        _ = ButlerStandardizer(DatasetId(3), butler=self.butler)

        _ = Standardizer.get(DatasetRef(DatasetId(5)), butler=self.butler)
        _ = Standardizer.get(DatasetId(6), butler=self.butler)

        _ = Standardizer.get(DatasetId(6), butler=self.butler, force=ButlerStandardizer)

    def test_standardize(self):
        """Test ButlerStandardizer instantiates and standardizes as expected."""
        std = Standardizer.get(DatasetId(7, fill_metadata=True), butler=self.butler)
        standardized = std.standardize()

        fits = FitsFactory.get_fits(7, spoof_data=True)
        hdr = fits["PRIMARY"].header
        expected = {
            "mjd": Time(hdr["DATE-AVG"], format="isot").mjd,
            "filter": hdr["FILTER"],
            "dataId": "7",
            "visit": hdr["EXPID"],
            "location": "file://far/far/away",
        }

        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                # mjd is almost eqaul, sometimes we offset to middle of exposure
                if k == "mjd":
                    self.assertAlmostEqual(v, standardized["meta"][k], 2)
                else:
                    self.assertEqual(v, standardized["meta"][k])

        # The CRVAL1/2 are with respect to the origin (CRPIX), Our center_ra
        # definition uses the pixel in the center of the CCD. The permissible
        # deviation should be on the scale of half a CCD's footprint, unless
        # it's DECam then it could be as big as half an FOV of the focal plane
        self.assertAlmostEqual(standardized["meta"]["ra"], fits[1].header["CRVAL1"], 1)
        self.assertAlmostEqual(standardized["meta"]["dec"], fits[1].header["CRVAL2"], 1)

        # compare standardized images
        # fmt: off
        np.testing.assert_equal([fits["IMAGE"].data,], standardized["science"])
        np.testing.assert_equal([fits["VARIANCE"].data,], standardized["variance"])
        np.testing.assert_equal([fits["MASK"].data,], standardized["mask"])
        # fmt: on

        # these are not easily comparable so just assert they exist
        self.assertTrue(standardized["meta"]["wcs"])
        self.assertTrue(standardized["meta"]["bbox"])

    def test_roundtrip(self):
        """Test ButlerStandardizer can instantiate itself from standardized
        data and a Data Butler."""
        std = Standardizer.get(DatasetId(8), butler=self.butler)
        standardized = std.standardize()

        std2 = ButlerStandardizer(**standardized["meta"], butler=self.butler)
        self.assertIsInstance(std2, ButlerStandardizer)

        standardized2 = std2.standardize()
        # TODO: I got to come up with some reasonable way of comparing this
        for k in [
            "location",
            "bbox",
            "mjd",
            "filter",
            "dataId",
            "OBSID",
            "ra",
            "dec",
            "visit",
            "filter",
            "detector",
        ]:
            with self.subTest("Failed to rounndtrip", key=k):
                self.assertEqual(standardized["meta"][k], standardized2["meta"][k])

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
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_bitmasking)

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
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_thresholding)

        conf = StandardizerConfig(
            {
                "grow_mask": False,
                "do_threshold": True,
                "brightness_threshold": 2,
            }
        )
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
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_growmask)

        conf = StandardizerConfig({"grow_mask": True, "grow_kernel_shape": (3, 3)})
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

    def test_to_layered_image(self):
        """Test ButlerStandardizer can create a LayeredImage."""
        std = Standardizer.get(DatasetId(8), butler=self.butler)
        self.assertIsInstance(std, ButlerStandardizer)

        # Get the expected FITS files and extract the MJD from the header
        fits = FitsFactory.get_fits(8, spoof_data=True)
        hdr = fits["PRIMARY"].header
        expected_mjd = Time(hdr["DATE-AVG"]).mjd

        # Get list of layered images froom the standardizer
        butler_imgs = std.toLayeredImage()
        self.assertEqual(1, len(butler_imgs))
        img = butler_imgs[0]

        # Compare standardized images
        np.testing.assert_equal(fits["IMAGE"].data, img.get_science().image)
        np.testing.assert_equal(fits["VARIANCE"].data, img.get_variance().image)
        np.testing.assert_equal(fits["MASK"].data, img.get_mask().image)

        # Test that we correctly set metadata
        # times can only be compred approximately, because sometimes we
        # calculate the time in the middle of the exposure
        self.assertAlmostEqual(expected_mjd, img.get_obstime(), 2)
        self.assertAlmostEqual(expected_mjd, img.get_science().obstime, 2)
        self.assertAlmostEqual(expected_mjd, img.get_variance().obstime, 2)
        self.assertAlmostEqual(expected_mjd, img.get_mask().obstime, 2)


if __name__ == "__main__":
    unittest.main()
