import unittest

import astropy.coordinates
import astropy.units
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

    def test_wcs_from_dict(self):
        # The base dictionary is good.
        self.assertIsNotNone(wcs_from_dict(self.header_dict))

        # Remove a required word and fail.
        del self.header_dict["CRVAL1"]
        self.assertIsNone(wcs_from_dict(self.header_dict))

    def test_extract_wcs_from_hdu_header(self):
        # The base dictionary is good.
        self.assertIsNotNone(extract_wcs_from_hdu_header(self.header))

        # Remove a required word and fail.
        del self.header["CRVAL1"]
        self.assertIsNone(extract_wcs_from_hdu_header(self.header))

    def test_wcs_to_dict(self):
        new_dict = wcs_to_dict(self.wcs)
        for key in self.header_dict:
            self.assertTrue(key in new_dict)
            self.assertAlmostEqual(new_dict[key], self.header_dict[key])

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

    def test_make_fake_wcs_info(self):
        # Test that we make the dictionary
        wcs_dict = make_fake_wcs_info(25.0, -10.0, 200, 100, deg_per_pixel=0.01)
        self.assertEqual(wcs_dict["WCSAXES"], 2)
        self.assertEqual(wcs_dict["CTYPE1"], "RA---TAN-SIP")
        self.assertEqual(wcs_dict["CTYPE2"], "DEC--TAN-SIP")
        self.assertEqual(wcs_dict["CRVAL1"], 25.0)
        self.assertEqual(wcs_dict["CRVAL2"], -10.0)
        self.assertEqual(wcs_dict["CRPIX1"], 100.0)
        self.assertEqual(wcs_dict["CRPIX2"], 50.0)
        self.assertEqual(wcs_dict["CDELT1"], 0.01)
        self.assertEqual(wcs_dict["CDELT2"], 0.01)
        self.assertEqual(wcs_dict["CTYPE1A"], "LINEAR  ")
        self.assertEqual(wcs_dict["CTYPE2A"], "LINEAR  ")
        self.assertEqual(wcs_dict["CUNIT1A"], "PIXEL   ")
        self.assertEqual(wcs_dict["CUNIT2A"], "PIXEL   ")

        # Test the we can convert to a WCS and extra predictions.
        test_wcs = WCS(wcs_dict)

        # Center position should be at approximately (25.0, -10.0).
        pos = test_wcs.pixel_to_world(99, 49)
        self.assertAlmostEqual(pos.ra.degree, 25.0, delta=0.001)
        self.assertAlmostEqual(pos.dec.degree, -10.0, delta=0.001)

        # One pixel off position should be at approximately (25.01, -10.0).
        pos = test_wcs.pixel_to_world(100, 48)
        self.assertAlmostEqual(pos.ra.degree, 25.01, delta=0.01)
        self.assertAlmostEqual(pos.dec.degree, -10.0, delta=0.01)


class test_construct_wcs_tangent_projection(unittest.TestCase):
    def test_requires_parameters(self):
        with self.assertRaises(TypeError):
            wcs = construct_wcs_tangent_projection()

    def test_only_required_parameter(self):
        wcs = construct_wcs_tangent_projection(None)
        self.assertIsNotNone(wcs)

    def test_one_pixel(self):
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = construct_wcs_tangent_projection(ref_val, img_shape=[1, 1], image_fov=3.5 * astropy.units.deg)
        self.assertIsNotNone(wcs)
        skyval = wcs.pixel_to_world(0, 0)
        refsep = ref_val.separation(skyval).to(astropy.units.deg).value
        self.assertAlmostEqual(0, refsep, places=5)

    def test_two_pixel(self):
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = construct_wcs_tangent_projection(ref_val, img_shape=[2, 2], image_fov=3.5 * astropy.units.deg)
        self.assertIsNotNone(wcs)
        skyval = wcs.pixel_to_world(0.5, 0.5)
        refsep = ref_val.separation(skyval).to(astropy.units.deg).value
        self.assertAlmostEqual(0, refsep, places=5)

    def test_image_field_of_view(self):
        """Test that the image field of view can be set explicitly."""
        fov_wanted = 3.5 * astropy.units.deg
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = construct_wcs_tangent_projection(
            ref_val, img_shape=[16, 16], image_fov=fov_wanted, solve_for_image_fov=True
        )
        self.assertIsNotNone(wcs)
        fov_actual = calc_actual_image_fov(wcs)[0]
        self.assertAlmostEqual(fov_wanted.value, fov_actual.value, places=8)

    def test_image_field_of_view_wide(self):
        """Test that the image field of view measured
        off the image returned expected values.
        """
        fov_wanted = [30.0, 15.0] * astropy.units.deg
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = construct_wcs_tangent_projection(
            ref_val, img_shape=[64, 32], image_fov=fov_wanted[0], solve_for_image_fov=True
        )
        self.assertIsNotNone(wcs)
        fov_actual = calc_actual_image_fov(wcs)
        self.assertAlmostEqual(fov_wanted[0].value, fov_actual[0].value, places=8)
        self.assertAlmostEqual(fov_wanted[1].value, fov_actual[1].value, places=8)

    def test_image_field_of_view_tall(self):
        """Test that the image field of view measured
        off the image returned expected values.
        """
        fov_wanted = [15.0, 29.05191311] * astropy.units.deg
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = construct_wcs_tangent_projection(
            ref_val, img_shape=[32, 64], image_fov=fov_wanted[0], solve_for_image_fov=True
        )
        self.assertIsNotNone(wcs)
        fov_actual = calc_actual_image_fov(wcs)
        self.assertAlmostEqual(fov_wanted[0].value, fov_actual[0].value, places=8)
        self.assertAlmostEqual(fov_wanted[1].value, fov_actual[1].value, places=8)


if __name__ == "__main__":
    unittest.main()
