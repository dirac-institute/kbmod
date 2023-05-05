import unittest
import kbmod.analysis.wcs_utils
import astropy.coordinates
import astropy.units


class test_construct_wcs_tangent_projection(unittest.TestCase):
    def test_requires_parameters(self):
        with self.assertRaises(TypeError):
            wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection()

    def test_only_required_parameter(self):
        wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection(None)
        self.assertIsNotNone(wcs)

    def test_one_pixel(self):
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection(
            ref_val, img_shape=[1, 1], image_fov=6.1 * astropy.units.deg
        )
        self.assertIsNotNone(wcs)
        skyval = wcs.pixel_to_world(0, 0)
        refsep = ref_val.separation(skyval).to(astropy.units.deg).value
        self.assertAlmostEqual(0, refsep, places=5)

    def test_two_pixel(self):
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection(
            ref_val, img_shape=[2, 2], image_fov=6.1 * astropy.units.deg
        )
        self.assertIsNotNone(wcs)
        skyval = wcs.pixel_to_world(0.5, 0.5)
        refsep = ref_val.separation(skyval).to(astropy.units.deg).value
        self.assertAlmostEqual(0, refsep, places=5)
