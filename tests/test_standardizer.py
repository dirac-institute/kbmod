import unittest
import tempfile
import warnings
import os

from astropy.io import fits as fitsio
from astropy.time import Time
import numpy as np

from utils import DECamImdiffFactory
from kbmod import PSF, Standardizer, StandardizerConfig
from kbmod.standardizers import (
    KBMODV1,
    KBMODV1Config,
    FitsStandardizer,
)


# Use a shared factory to skip having to untar the archive
FitsFactory = DECamImdiffFactory()


class MyStd(KBMODV1):
    """Custom standardizer for testing Standardizer registration"""

    name = "MyStd"
    priority = 3
    testing_kwargs = False

    @classmethod
    def yesStandardize(cls, tgt):
        _, resources = super().resolveTarget(tgt)
        return True, resources

    @classmethod
    def noStandardize(cls, tgt):
        return False, {}

    @classmethod
    def resolveTarget(cls, tgt):
        return cls.noStandardize(tgt)

    def __init__(self, *args, required_flag, optional_flag=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.testing_kwargs:
            self.required_flag = False
        else:
            self.required_flag = required_flag
        self.optional_flag = optional_flag

    def translateHeader(self):
        # invoke the parent functionality to standardize the default values
        metadata = super().translateHeader()
        metadata["required_flag"] = self.required_flag
        if self.optional_flag:
            metadata["optional_flag"] = self.optional_flag
        return metadata


class TestStandardizer(unittest.TestCase):
    """Test Standardizer class."""

    def setUp(self):
        self.fits = FitsFactory.mock_fits()
        # Ignore user warning about multiple standardizers,
        # One of them will be the MyStd
        warnings.filterwarnings(action="ignore", category=UserWarning, message="Multiple standardizers")

    def tearDown(self):
        # restore defaults
        MyStd.resolveTarget = MyStd.noStandardize
        MyStd.priority = 3
        warnings.resetwarnings()

    def test_kwargs_to_init(self):
        """Test kwargs are correctly passed from top-level Standardizer to the
        underlying standardizer implementation."""
        MyStd.resolveTarget = MyStd.yesStandardize
        MyStd.testing_kwargs = True

        with self.assertRaises(TypeError):
            std = Standardizer.get(self.fits)

        with self.assertWarnsRegex(UserWarning, "Multiple standardizers"):
            std = Standardizer.get(self.fits, required_flag=False)
        stdmeta = std.standardizeMetadata()
        self.assertFalse(stdmeta["required_flag"])

        std = Standardizer.get(self.fits, required_flag=True, optional_flag=True)
        stdmeta = std.standardizeMetadata()
        self.assertTrue(stdmeta["required_flag"])
        self.assertIn("optional_flag", stdmeta)
        self.assertTrue(stdmeta["optional_flag"])

    def test_instantiation(self):
        """Test priority, forcing and automatic selection works."""
        std = Standardizer.get(self.fits, required_flag=True)
        self.assertIsInstance(std, KBMODV1)

        MyStd.resolveTarget = MyStd.yesStandardize
        std = Standardizer.get(self.fits, required_flag=True)
        self.assertIsInstance(std, MyStd)

        MyStd.priority = 0
        std = Standardizer.get(self.fits, required_flag=True)
        self.assertIsInstance(std, KBMODV1)

        # Test forcing ignores everything
        MyStd.resolveTarget = MyStd.noStandardize
        MyStd.priority = 0
        std = Standardizer.get(self.fits, force=MyStd, required_flag=True)
        self.assertIsInstance(std, MyStd)

        # Test instantiating from a single HDUList
        std = Standardizer.get(self.fits)
        self.assertIsInstance(std, Standardizer)

        # Test force direct and named
        std2 = Standardizer.get(self.fits, force=KBMODV1)
        self.assertIsInstance(std, KBMODV1)
        self.assertEqual(std2.location, std.location)
        self.assertEqual(std2.hdulist, std.hdulist)

        std2 = Standardizer.get(self.fits, force="KBMODV1")
        self.assertIsInstance(std, KBMODV1)
        self.assertEqual(std2.location, std.location)
        self.assertEqual(std2.hdulist, std.hdulist)

        # Test from path
        hdul = FitsFactory.mock_fits(spoof_data=True)
        tmpf = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        hdul.writeto(tmpf.file, overwrite=True)
        hdul.close()
        tmpf.close()

        std2 = Standardizer.get(tmpf.name)
        self.assertIsInstance(std, KBMODV1)
        self.assertEqual(std2.location, tmpf.name)
        self.assertEqual(len(std2.hdulist), 16)

        # clean up resources
        os.unlink(tmpf.name)


# This is in test_standardizeer because totest Standardizer because it's easier
# than making multiple new standardizers for sake of technical clarity or style
# Test KBMODV1 more extensively than other standardizers to cover for the
# possible code-paths through Standardizer itself. TODO: eventually update this
class TestKBMODV1(unittest.TestCase):
    """Test KBMODV1 Standardizer and Standardizer."""

    def setUp(self):
        self.fits = FitsFactory.mock_fits(spoof_data=True)

    def tearDown(self):
        self.fits.close(output_verify="ignore")

    def test_init_direct(self):
        # Test default values are as expected and that parent classes did their
        # share of work
        std = KBMODV1(hdulist=self.fits)
        self.assertEqual(std.location, ":memory:")
        self.assertEqual(std.hdulist, self.fits)
        self.assertEqual(std.config, KBMODV1Config())
        self.assertEqual(std.primary, self.fits["PRIMARY"].header)
        self.assertEqual(
            std.processable,
            [
                self.fits["IMAGE"],
            ],
        )
        self.assertTrue(std.isMultiExt)
        self.assertTrue(KBMODV1.canStandardize(self.fits))

        # Test init from filepath
        # Test from path
        hdul = FitsFactory.mock_fits(spoof_data=True)
        fits_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        hdul.writeto(fits_file.file, overwrite=True)
        hdul.close()
        fits_file.close()

        # Test init from HDUList with a known location
        hdul = fitsio.open(fits_file.name)
        std = KBMODV1(hdulist=hdul)
        self.assertEqual(std.location, fits_file.name)

        # Test init with both
        std2 = KBMODV1(location=fits_file.name, hdulist=hdul)
        self.assertEqual(std.location, fits_file.name)
        self.assertEqual(std.hdulist, std2.hdulist)

        # Test raises when neither
        with self.assertRaisesRegex(ValueError, "Expected location or HDUList"):
            KBMODV1()

        # Test raises correctly when location makes no sense
        with self.assertRaisesRegex(FileNotFoundError, "location is not a file"):
            KBMODV1("noexist", hdulist=hdul)
            KBMODV1("noexist")

        # clean up resources
        os.unlink(fits_file.name)

    def test_standardization(self):
        """Test KBMODV1 standardize executes and standardizes metadata."""
        std = Standardizer.get(self.fits, force=KBMODV1)
        standardized = std.standardize()

        for key in ["meta", "science", "mask", "variance", "psf"]:
            self.assertIn(key, standardized.keys())

        hdr = self.fits["PRIMARY"].header
        expected = {
            "mjd_mid": Time(hdr["DATE-AVG"], format="isot").mjd
            + (hdr["EXPREQ"] + 0.5) / 2.0 / 60.0 / 60.0 / 24.0,
            "obs_lat": hdr["OBS-LAT"],
            "obs_lon": hdr["OBS-LONG"],
            "obs_elev": hdr["OBS-ELEV"],
            "location": ":memory:",
            "FILTER": hdr["FILTER"],
            "IDNUM": hdr["IDNUM"],
            "OBSID": hdr["OBSID"],
            "DTNSANAM": hdr["DTNSANAM"],
            "AIRMASS": hdr["AIRMASS"],
            "GAINA": hdr["GAINA"],
            "GAINB": hdr["GAINB"],
        }

        # There used to be an assertDictContainsSubset, but got deprecated?
        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                self.assertEqual(v, standardized["meta"][k])

        # consequence of making std methods generators is that they need to be
        # evaluated, see kbmov1.py, perhaps we should give up on this?
        empty_array = np.zeros((5, 5), np.float32)
        np.testing.assert_equal(empty_array, next(standardized["science"]))
        np.testing.assert_equal(empty_array, next(standardized["variance"]))
        np.testing.assert_equal(empty_array.astype(np.int32), next(standardized["mask"]))

    def test_roundtrip(self):
        """Test KBMODV1 can instantiate itself from standardized data."""
        std = Standardizer.get(self.fits, force=KBMODV1)
        standardized = std.standardize()

        # Test it raises correctly when file is not on disk
        with self.assertRaisesRegex(FileNotFoundError, "location is not a file, but no hdulist"):
            KBMODV1(**standardized["meta"], force=KBMODV1)

        # Test it works correctly when the FITS is reachable.
        fits_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        self.fits.writeto(fits_file.file, overwrite=True, output_verify="ignore")
        fits_file.close()

        std = Standardizer.get(fits_file.name, force=KBMODV1)
        standardized = std.standardize()
        std2 = KBMODV1(**standardized["meta"])
        self.assertIsInstance(std2, KBMODV1)

    def test_bitmasking(self):
        """Test masking with direct config works as expected."""
        # Assign each flag that exists to a pixel, standardize, then expect
        # the mask to only contain those pixels that are also in mask_flags.
        # The grow_kernel is so large by default it would mask the nearly the
        # whole image, so we turn it off.
        KBMODV1Config.grow_mask = False
        mask_arr = self.fits["MASK"].data
        for i, flag in enumerate(KBMODV1Config.bit_flag_map):
            mask_arr.ravel()[i] = KBMODV1Config.bit_flag_map[flag]

        std = Standardizer.get(self.fits, force=KBMODV1)
        standardizedMask = std.standardizeMaskImage()

        for mask in standardizedMask:
            for i, flag in enumerate(KBMODV1Config.bit_flag_map):
                with self.subTest("Failed to mask expected", flag=flag):
                    if flag in KBMODV1Config.mask_flags:
                        self.assertEqual(mask.ravel()[i], True)
                    else:
                        self.assertEqual(mask.ravel()[i], False)

    def test_threshold_masking(self):
        """Test brightness threshold masking. Test config overrides."""
        # set one pixel that is masked and one that isn't
        self.fits["IMAGE"].data[1, 1] = 1
        self.fits["IMAGE"].data[2, 2] = 3

        conf = StandardizerConfig(
            {
                "grow_mask": False,
                "do_threshold": True,
                "brightness_threshold": 2,
            }
        )
        std = Standardizer.get(self.fits, force=KBMODV1, config=conf)
        mask = next(std.standardizeMaskImage())

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[2, 2])

    def test_grow_mask(self):
        """Test mask grows as expected."""
        # set central pixel to be masked, then grow that mask to all its
        # neighbors.
        self.fits["MASK"].data[2, 2] = KBMODV1Config.bit_flag_map["BAD"]

        conf = StandardizerConfig({"grow_mask": True, "grow_kernel_shape": (3, 3)})
        std = Standardizer.get(self.fits, force=KBMODV1, config=conf)
        mask = next(std.standardizeMaskImage())

        # Note that this is different than masking via Manhattan neighbors -
        # which can be implemented by using the C++ functions in KBMODV1, do I?
        # the solution now is an masked square in the center of the array
        self.assertTrue(mask[1:3, 1:3].all())
        self.assertFalse(mask[:, 0].all())
        self.assertFalse(mask[0, :].all())
        self.assertFalse(mask[-1, :].all())
        self.assertFalse(mask[:, -1].all())

    def test_psf(self):
        """Test PSFs are created as expected. Test instance config overrides."""
        std = Standardizer.get(self.fits, force=KBMODV1)

        psf = next(std.standardizePSF())
        self.assertIsInstance(psf, PSF)
        self.assertEqual(psf.get_std(), std.config["psf_std"])

        std.config["psf_std"] = 2
        psf = next(std.standardizePSF())
        self.assertIsInstance(psf, PSF)
        self.assertEqual(psf.get_std(), std.config["psf_std"])

        # make sure we didn't override any of the global defaults by accident
        std2 = Standardizer.get(self.fits, force=KBMODV1)
        self.assertNotEqual(std2.config, std.config)

        # Test iterable PSF STD configuration
        std2.config["psf_std"] = [
            3,
        ]
        psf = next(std2.standardizePSF())
        self.assertEqual(psf.get_std(), std2.config["psf_std"][0])

    def test_to_layered_image(self):
        """Test that KBMODV1 standardizer can create LayeredImages."""
        conf = KBMODV1Config({"greedy_export": True})
        std = Standardizer.get(self.fits, force=KBMODV1, config=conf)
        self.assertIsInstance(std, KBMODV1)

        # Get the expected FITS files and extract the MJD from the header
        hdr = self.fits["PRIMARY"].header
        offset_to_mid = (hdr["EXPREQ"] + 0.5) / 2.0 / 60.0 / 60.0 / 24.0
        expected_mjd = Time(hdr["DATE-AVG"], format="isot").mjd + offset_to_mid

        # Get list of layered images from the standardizer
        layered_imgs = std.toLayeredImage()
        self.assertEqual(1, len(layered_imgs))
        img = layered_imgs[0]

        # Compare standardized images
        np.testing.assert_equal(self.fits["IMAGE"].data, img.get_science().image)
        np.testing.assert_equal(self.fits["VARIANCE"].data, img.get_variance().image)
        np.testing.assert_equal(self.fits["MASK"].data, img.get_mask().image)

        # Test that we correctly set metadata
        self.assertEqual(expected_mjd, img.get_obstime())

    def test_to_layered_image_no_greedy(self):
        """Test that KBMODV1 standardizer can create LayeredImages. Explicitly
        setting `greedy_export` to False, which is the default."""
        conf = KBMODV1Config({"greedy_export": False})
        std = Standardizer.get(self.fits, force=KBMODV1, config=conf)
        self.assertIsInstance(std, KBMODV1)

        # Get the expected FITS files and extract the MJD from the header
        hdr = self.fits["PRIMARY"].header
        offset_to_mid = (hdr["EXPREQ"] + 0.5) / 2.0 / 60.0 / 60.0 / 24.0
        expected_mjd = Time(hdr["DATE-AVG"], format="isot").mjd + offset_to_mid

        # Get list of layered images from the standardizer
        layered_imgs = std.toLayeredImage()
        self.assertEqual(1, len(layered_imgs))
        img = layered_imgs[0]

        # Assert that "IMAGE" data is None, but we do not check VARIANCE or MASK,
        # because in the KBMODV1 standardizer, those are not set to None considered
        # processable. See the definition of `self.processable` in the __init__
        # method of KBMODV1.
        self.assertIsNone(self.fits["IMAGE"].data)

        # Test that we correctly set metadata
        self.assertEqual(expected_mjd, img.get_obstime())

if __name__ == "__main__":
    unittest.main()
