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
            ref_val, img_shape=[1, 1], image_fov=3.5 * astropy.units.deg
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
            ref_val, img_shape=[2, 2], image_fov=3.5 * astropy.units.deg
        )
        self.assertIsNotNone(wcs)
        skyval = wcs.pixel_to_world(0.5, 0.5)
        refsep = ref_val.separation(skyval).to(astropy.units.deg).value
        self.assertAlmostEqual(0, refsep, places=5)

    def test_image_field_of_view(self):
        """Test that the image field of view can be set explicitly.
        """
        fov_wanted = 3.5 * astropy.units.deg
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection(
            ref_val, img_shape=[16, 16], image_fov=fov_wanted,
            solve_for_image_fov=True
        )
        self.assertIsNotNone(wcs)
        fov_actual = kbmod.analysis.wcs_utils.calc_actual_image_fov(wcs)[0]
        self.assertAlmostEqual(fov_wanted.value, fov_actual.value, places=8)
    
    def test_image_field_of_view_wide(self):
        """Test that the image field of view measured
        off the image returned expected values.
        """
        fov_wanted = [ 30.0, 15.0 ] * astropy.units.deg
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection(
            ref_val, img_shape=[64, 32], image_fov=fov_wanted[0],
            solve_for_image_fov=True
        )
        self.assertIsNotNone(wcs)
        fov_actual = kbmod.analysis.wcs_utils.calc_actual_image_fov(wcs)
        self.assertAlmostEqual(fov_wanted[0].value, fov_actual[0].value, places=8)
        self.assertAlmostEqual(fov_wanted[1].value, fov_actual[1].value, places=8)
    
    def test_image_field_of_view_tall(self):
        """Test that the image field of view measured
        off the image returned expected values.
        """
        fov_wanted = [ 15.0, 29.05191311 ] * astropy.units.deg
        ref_val = astropy.coordinates.SkyCoord(
            ra=0 * astropy.units.deg, dec=0 * astropy.units.deg, frame="icrs"
        )
        wcs = kbmod.analysis.wcs_utils.construct_wcs_tangent_projection(
            ref_val, img_shape=[32, 64], image_fov=fov_wanted[0],
            solve_for_image_fov=True
        )
        self.assertIsNotNone(wcs)
        fov_actual = kbmod.analysis.wcs_utils.calc_actual_image_fov(wcs)
        self.assertAlmostEqual(fov_wanted[0].value, fov_actual[0].value, places=8)
        self.assertAlmostEqual(fov_wanted[1].value, fov_actual[1].value, places=8)
