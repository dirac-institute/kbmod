import unittest

from astropy.wcs import WCS
from astropy.io import fits

from kbmod.wcs_utils import *


class test_wcs_conversion(unittest.TestCase):
    def setUp(self):
        self.header_dict = {
            "WCSAXES": 2,
            "CTYPE1": "RA---TAN-SIP",
            "CTYPE2": "DEC--TAN-SIP",
            "CRVAL1": 200.614997245422,
            "CRVAL2": -7.78878863332778,
            "CRPIX1": 1033.934327,
            "CRPIX2": 2043.548284,
        }
        self.wcs = WCS(self.header_dict)
        self.header = self.wcs.to_header()

    def test_wcs_equal_fits(self):
        self.assertTrue(wcs_fits_equal(self.wcs, self.wcs))
        self.assertTrue(wcs_fits_equal(None, None))
        self.assertFalse(wcs_fits_equal(None, self.wcs))
        self.assertFalse(wcs_fits_equal(self.wcs, None))

        self.header_dict["CRVAL1"] = 201.5
        wcs2 = WCS(self.header_dict)
        self.assertFalse(wcs_fits_equal(self.wcs, wcs2))

        wcs3 = WCS(self.header_dict)
        self.assertTrue(wcs_fits_equal(wcs2, wcs3))

    def test_extract_wcs_from_hdu_header(self):
        # The base dictionary is good.
        self.assertIsNotNone(extract_wcs_from_hdu_header(self.header))

        # Remove a required word and fail.
        del self.header["CRVAL1"]
        self.assertIsNone(extract_wcs_from_hdu_header(self.header))

    def test_serialization(self):
        self.wcs.pixel_shape = (200, 250)
        wcs_str = serialize_wcs(self.wcs)
        self.assertTrue(isinstance(wcs_str, str))

        wcs2 = deserialize_wcs(wcs_str)
        self.assertTrue(isinstance(wcs2, WCS))
        self.assertEqual(self.wcs.pixel_shape, wcs2.pixel_shape)
        self.assertTrue(wcs_fits_equal(self.wcs, wcs2))

        # Test that we can serialize and deserialize None.
        none_str = serialize_wcs(None)
        self.assertEqual(none_str, "")
        self.assertIsNone(deserialize_wcs(""))
        self.assertIsNone(deserialize_wcs("none"))
        self.assertIsNone(deserialize_wcs("None"))

    def test_append_wcs_to_hdu_header(self):
        for use_dictionary in [True, False]:
            if use_dictionary:
                wcs_info = self.header_dict
            else:
                wcs_info = self.wcs

            # Run the dictionary and full WCS tests as separate subtests
            with self.subTest(i=use_dictionary):
                pri = fits.PrimaryHDU()
                self.assertFalse("CRVAL1" in pri.header)
                self.assertFalse("CRVAL2" in pri.header)
                self.assertFalse("CRPIX1" in pri.header)
                self.assertFalse("CRPIX2" in pri.header)

                append_wcs_to_hdu_header(wcs_info, pri.header)
                for key in self.header_dict:
                    self.assertTrue(key in pri.header)
                    self.assertAlmostEqual(pri.header[key], self.header_dict[key])

    def test_make_fake_wcs(self):
        # Test that we can convert to a WCS and perform predictions.
        test_wcs = make_fake_wcs(25.0, -10.0, 200, 100, deg_per_pixel=0.01)

        # Center position should be at approximately (25.0, -10.0).
        pos = test_wcs.pixel_to_world(99, 49)
        self.assertAlmostEqual(pos.ra.degree, 25.0, delta=0.001)
        self.assertAlmostEqual(pos.dec.degree, -10.0, delta=0.001)

        # One pixel off position should be at approximately (25.01, -10.0).
        pos = test_wcs.pixel_to_world(100, 48)
        self.assertAlmostEqual(pos.ra.degree, 25.01, delta=0.01)
        self.assertAlmostEqual(pos.dec.degree, -10.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
