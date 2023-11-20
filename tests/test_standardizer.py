import unittest
import tempfile
import warnings
import os

from astropy.io import fits
from astropy.time import Time
import numpy as np

from utils import DECamImdiffFactory
from kbmod import PSF, Standardizer, StandardizerConfig
from kbmod.standardizers import (KBMODV1,
                                 KBMODV1Config,
                                 FitsStandardizer,
                                 SingleExtensionFits,
                                 MultiExtensionFits)


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
        metadata["required_flag"] = False
        if self.required_flag:
            metadata["required_flag"] = True
        if self.optional_flag:
            metadata["optional_flag"] = True
        return metadata

class TestStandardizer(unittest.TestCase):
    """Test Standardizer class."""
    def setUp(self):
        self.fits = FitsFactory.mock_fits()
        empty_array = np.zeros((5, 5), np.float32)
        self.fits["IMAGE"].data = empty_array
        self.fits["VARIANCE"].data = empty_array
        self.fits["MASK"].data = empty_array.astype(np.int32)
        self.img = empty_array
        self.mask = empty_array.astype(int)
        # ignore multiple volunteered standardizer warnings
        warnings.filterwarnings("ignore",
                                message="Multiple standardizers declared",
                                category=UserWarning)

    def tearDown(self):
        # restore defaults
        MyStd.resolveTarget = MyStd.noStandardize
        MyStd.priority = 3
        warnings.resetwarnings()

        # release resources
        self.fits.close(output_verify="ignore")

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
        std = Standardizer.get(self.fits,  required_flag=True)
        self.assertIsInstance(std, KBMODV1)

        # Test forcing ignores everything
        MyStd.resolveTarget = MyStd.noStandardize
        MyStd.priority = 0
        std = Standardizer.get(self.fits, force=MyStd,
                               required_flag=True)
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

        # see comments in test_init_direct
        fits = FitsFactory.mock_fits()
        for hdu in fits[:4]:
            hdu.header["NAXIS"] = 0
            hdu.header.remove("NAXIS1", ignore_missing=True)
            hdu.header.remove("NAXIS2", ignore_missing=True)
        tmpf = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        fits[:4].writeto(tmpf.file, overwrite=True, output_verify="ignore")
        tmpf.close()

        std2 = Standardizer.get(tmpf.name)
        self.assertIsInstance(std, KBMODV1)
        self.assertEqual(std2.location, tmpf.name)
        self.assertEqual(len(std2.hdulist), 4)

        # clean up resources
        std2.close()
        os.unlink(tmpf.name)


# This is in test_standardizeer because totest Standardizer because it's easier
# than making multiple new standardizers for sake of technical clarity or style
# Test KBMODV1 more extensively than other standardizers to cover for the
# possible code-paths through Standardizer itself. TODO: eventually update this
class TestKBMODV1(unittest.TestCase):
    """Test KBMODV1 Standardizer and Standardizer."""

    def setUp(self):
        self.fits = FitsFactory.mock_fits()

        empty_array = np.zeros((5, 5), np.float32)
        self.fits["IMAGE"].data = empty_array
        self.fits["VARIANCE"].data = empty_array
        self.fits["MASK"].data = empty_array.astype(np.int32)

        self.img = empty_array
        self.mask = empty_array.astype(int)

    def tearDown(self):
        # Note that np.int32 in setUp is necessary because astropy will raise
        # throw a RuntimeError here otherwise. If we gave it a CompImageHDU, it
        # expects an 32bit int as data and can't handle getting anything else
        # See: https://docs.astropy.org/en/stable/io/fits/usage/image.html
        self.fits.close(output_verify="ignore")

    def mock_fits_file(self):
        # We only mock the headers. AstroPy can not read an empty bintable
        # because it can not construct byte offsets. We can not make one up,
        # because an error is raised if data layout doesn't match the header's
        # description of it. We know resolveTarget and .standardize will not
        # look at anything but the first 4 HDUs for this class, so we cheat and
        # make it look empty. Ugly and fragile but the only other option is to
        # store the actual files. Insult to injury, the delete_on_close key
        # isn't available until Python 3.12
        fits = FitsFactory.mock_fits()
        for hdu in fits[:4]:
            hdu.header["NAXIS"] = 0
            hdu.header.remove("NAXIS1", ignore_missing=True)
            hdu.header.remove("NAXIS2", ignore_missing=True)

        tmpf = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        fits[:4].writeto(tmpf.file, overwrite=True, output_verify="ignore")
        tmpf.close()

        return tmpf

    def test_init_direct(self):
        # Test default values are as expected and that parent classes did their
        # share of work
        std = KBMODV1(self.fits)
        self.assertEqual(std.location, ":memory:")
        self.assertEqual(std.hdulist, self.fits)
        self.assertEqual(std.config, KBMODV1Config())
        self.assertEqual(std.primary, self.fits["PRIMARY"].header)
        self.assertEqual(std.processable, [self.fits["IMAGE"], ])
        self.assertTrue(std.isMultiExt)
        self.assertTrue(KBMODV1.canStandardize(self.fits))

        # Test init from filepath
        fits_file = self.mock_fits_file()
        std = KBMODV1(fits_file.name)
        self.assertEqual(std.location, fits_file.name)

        # Test init from HDUList with a known location
        f = fits.open(fits_file.name)
        std = KBMODV1(f)
        self.assertEqual(std.location, fits_file.name)

        # Test resources shortcut, doesn't really make sense, but is secretly
        # only used by Standardizer.get anyhow...
        std2 = KBMODV1(f, hdulist=f)
        self.assertEqual(std.location, fits_file.name)
        self.assertEqual(std.hdulist, std2.hdulist)

        # Test raises correctly when location makes no sense
        with self.assertRaisesRegex(FileNotFoundError, "location is not a file"):
            KBMODV1("noexist", hdulist=f)

        # Test raises when tgt isn't an path or an HDUList
        with self.assertRaisesRegex(TypeError, "Expected location or HDUList"):
            KBMODV1(fits_file)

        os.unlink(fits_file.name)

    def test_standardization(self):
        """Test KBMODV1 standardize executes and standardizes metadata."""
        std = Standardizer.get(self.fits, force=KBMODV1)
        standardized = std.standardize()

        for key in ["meta", "science", "mask", "variance", "psf"]:
            self.assertIn(key, standardized.keys())

        hdr = self.fits["PRIMARY"].header
        expected = {
            "mjd": Time(hdr["DATE-AVG"], format="isot").mjd,
            "filter": hdr["FILTER"],
            "visit_id": hdr["IDNUM"],
            "observat": hdr["OBSERVAT"],
            "obs_lat": hdr["OBS-LAT"],
            "obs_lon": hdr["OBS-LONG"],
            "obs_elev": hdr["OBS-ELEV"],
            "location": ":memory:"
            }

        # There used to be an assertDictContainsSubset, but got deprecated?
        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                self.assertEqual(v, standardized["meta"][k])

        # consequence of making std methods generators is that they need to be
        # evaluated, see kbmov1.py, perhaps we should give up on this?
        np.testing.assert_equal(self.img, next(standardized["science"]))
        np.testing.assert_equal(self.variance, next(standardized["variance"]))
        np.testing.assert_equal(self.mask, next(standardized["mask"]))

        # these are not easily comparable because they are fits file dependent
        # so just assert they exist
        self.assertTrue(standardized["meta"]["wcs"])
        self.assertTrue(standardized["meta"]["bbox"])

    def test_bitmasking(self):
        """Test masking with direct config works as expected."""
        # Assign each flag that exists to a pixel, standardize, then expect
        # the mask to only contain those pixels that are also in mask_flags.
        # The grow_kernel is so large by default it would mask the nearly the
        # whole image, so we turn it off.
        KBMODV1Config.grow_mask = False
        mask_arr = self.mask
        for i, flag in enumerate(KBMODV1Config.bit_flag_map):
            mask_arr.ravel()[i] = KBMODV1Config.bit_flag_map[flag]

        # set the fits arrays
        self.fits["MASK"].data = mask_arr.astype(np.int32)

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

        conf = StandardizerConfig({
            "grow_mask": False,
            "do_threshold": True,
            "brightness_threshold": 2,
        })
        std = Standardizer.get(self.fits, force=KBMODV1,
                               config=conf)
        mask = next(std.standardizeMaskImage())

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[2, 2])

    def test_grow_mask(self):
        """Test mask grows as expected."""
        # set central pixel to be masked, then grow that mask to all its
        # neighbors.
        self.fits["MASK"].data[2, 2] = KBMODV1Config.bit_flag_map["BAD"]

        conf = StandardizerConfig({
            "grow_mask": True,
            "grow_kernel_shape": (3, 3)
        })
        std = Standardizer.get(self.fits, force=KBMODV1,
                               config=conf)
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
        std2.config["psf_std"] = [3, ]
        psf = next(std2.standardizePSF())
        self.assertEqual(psf.get_std(), std2.config["psf_std"][0])


if __name__ == "__main__":
    unittest.main()
