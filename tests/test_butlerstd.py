import os
import tempfile
import unittest
from unittest import mock

from astropy.time import Time
from astropy.wcs import WCS
import numpy as np

from utils import DECamImdiffFactory
from kbmod import PSF, Standardizer, StandardizerConfig
from kbmod.standardizers import KBMODV1Config


class Registry:
    def getDataset(self, returnvals):
        return returnvals

class MockButler:
    fitsFactory = DECamImdiffFactory()

    def __init__(self, root):
        self.root = root

    def getURI(self, ref, collections=None):
        mocked = mock.Mocked(name="ButlerURI")
        mocked.geturl.return_value = f"file:/{self.fitsFactory.groups[ref]['filename'][0]}"
        return mocked

    def getDataset(self, datid):
        return self.get(datid)

    def get(self, ref, collection=None):
        # We rely heavily on the fact that we know what exists in DECam headers
        hdul = self.fitsFactory.create_fits(ref % self.fitsFactory.n_files)
        prim = hdul["PRIMARY"].header

        # Butler get (the way we use it at least) is in the context of
        # returning an Exposure object. Exposure is like our LayeredImage. Now
        # we need to mock every attribute, method and property that we use in
        # the standardizer
        mocked = mock.Mock(
            name="Exposure",
            spec_set=["visitInfo", "info", "getWidth", "getHeight",
                      "image", "variance", "mask", "wcs"]
        )

        # General metadata mocks
        mocked.visitInfo.date.toAstropy.return_value = \
            Time(hdul["PRIMARY"].header["DATE-AVG"], format="isot")
        mocked.info.id = prim["EXPID"]
        mocked.getWidth.return_value = hdul[1].header["NAXIS1"]
        mocked.getHeight.return_value = hdul[1].header["NAXIS2"]
        mocked.getFilter().physicalLabel = prim["FILTER"]

        # WCS doesn't roundtrip NAXIS to/from headers and strings. Another
        # problem is that Rubin Sci. Pipes. return their own internal SkyWcs
        # object. Here we return a Header because that'll work with ButlerStd.
        # What happens if SkyWcs changes though?
        wcs = WCS(hdul[1].header).to_header()
        wcs["NAXIS1"] = prim["NAXIS1"]
        wcs["NAXIS2"] = prim["NAXIS2"]
        mock.hasWcs.return_value = True
        mocked.wcs.getFitsMetadata.return_value = wcs

        # Mocking the images
        empty_array = np.ones(5, 5, dtype=np.float32)
        mocked.image.array = empty_array
        mocked.variance.array = empty_array
        mocked.mask.array = empty_array.astype(np.int32)

        # Same issue as with WCS, what if/when the mask changes
        mocked.mask.getMaskPlaneDict.return_value = KBMODV1Config.bit_flag_map

        return mocked


class TestButlerStandardizer(unittest.TestCase):
    """Test KBMODV1 Standardizer. """

    def setUp(self):
        self.butler = MockButler()
        empty_array = np.ones(5, 5, dtype=np.float32)
        self.img = empty_array
        self.variance = empty_array
        self.mask = empty_array.astype(np.int32)

    def test_standardization(self):
        """Test ButlerStandardizer instantiates and executes as expected."""
        std = Standardizer
        std = Standardizer.fromHDUList(self.fits, forceStandardizer=KBMODV1)
        standardized = std.standardize()

        for key in ["meta", "science", "mask", "variance", "psf"]:
            self.assertIn(key, standardized.keys())

        hdr = self.fits["PRIMARY"].header
        expected = {
            'mjd': Time(hdr["DATE-AVG"], format="isot").mjd,
            'filter': hdr["FILTER"],
            'visit_id': hdr["IDNUM"],
            'observat': hdr["OBSERVAT"],
            'obs_lat': hdr["OBS-LAT"],
            'obs_lon': hdr["OBS-LONG"],
            'obs_elev': hdr["OBS-ELEV"],
            'location': ':memory:'
            }

        # There used to be an assertDictContainsSubset, but got deprecated?
        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                self.assertEqual(v, standardized["meta"][k])

        # consequence of making std methods generators is that they need to be
        # evaluated, see kbmov1.py, perhaps we should give up on this?
        np.testing.assert_equal(self.img, next(standardized["science"]))
        np.testing.assert_equal(self.img, next(standardized["variance"]))
        np.testing.assert_equal(self.mask, next(standardized["mask"]))

        # these are not easily comparable because they are fits file dependent
        # so just assert they exist
        self.assertTrue(standardized["meta"]["wcs"])
        self.assertTrue(standardized["meta"]["bbox"])

    def test_bitmasking(self):
        """Test masking with direct config works as expected."""
        # Assign each flag that exists to a pixel, standardize, then expect
        # the mask only masked the masked values and not the others
        # the grow_kernel is so large by default it would mask the nearly the
        # whole image, so we turn it off.
        KBMODV1Config.grow_mask = False
        mask_arr = self.mask
        for i, flag in enumerate(KBMODV1Config.bit_flag_map):
            mask_arr.ravel()[i] = KBMODV1Config.bit_flag_map[flag]

        # set the fits arrays
        self.fits["MASK"].data = mask_arr

        std = Standardizer.fromHDUList(self.fits, forceStandardizer=KBMODV1)
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
        std = Standardizer.fromHDUList(self.fits, forceStandardizer=KBMODV1,
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
        std = Standardizer.fromHDUList(self.fits, forceStandardizer=KBMODV1,
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
        std = Standardizer.fromHDUList(self.fits, forceStandardizer=KBMODV1)

        psf = next(std.standardizePSF())
        self.assertIsInstance(psf, PSF)
        self.assertEqual(psf.get_std(), std.config["psf_std"])

        std.config["psf_std"] = 2
        psf = next(std.standardizePSF())
        self.assertIsInstance(psf, PSF)
        self.assertEqual(psf.get_std(), std.config["psf_std"])

        # make sure we didn't override any of the global defaults by accident
        std2 = Standardizer.fromHDUList(self.fits, forceStandardizer=KBMODV1)
        self.assertNotEqual(std2.config, std.config)

        # Test iterable PSF STD configuration
        std2.config["psf_std"] = [3, ]
        psf = next(std2.standardizePSF())
        self.assertEqual(psf.get_std(), std2.config["psf_std"][0])


if __name__ == "__main__":
    unittest.main()
