import math
import unittest

import astropy.wcs
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import kbmod
import kbmod.analysis_utils
import kbmod.search


class test_calc_barycentric_corr(unittest.TestCase):
    def setUp(self):
        # Define the path for the data.
        im_filepath = "../data/demo"
        # The demo data has an object moving at x_v=10 px/day
        # and y_v = 0 px/day. So we search velocities [0, 20]
        # and angles [-0.5, 0.5].
        v_arr = [0, 20, 21]
        ang_arr = [0.5, 0.5, 11]

        self.input_parameters = {
            # Required
            "im_filepath": im_filepath,
            "res_filepath": None,
            "time_file": None,
            "output_suffix": "DEMO",
            "v_arr": v_arr,
            "ang_arr": ang_arr,
            "bary_dist": 50.0,
        }

    def _expected_string(self, baryCoeff, shape_coeff):
        """Return a string of the actual results in a form that can be copied into baryExpected in the test code. Used for as-built testing."""
        value_string = f"# fmt: off\nbaryExpected = np.array(["
        value_string += f"\n    #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,"
        for i in range(0, shape_coeff[0]):
            value_string += f"\n    [{baryCoeff[i,0]:23.16f}"
            for j in range(1, shape_coeff[1]):
                value_string += f", {baryCoeff[i,j]:23.16f}"
            value_string += f"],"
        value_string += f"\n]\n# fmt: on"
        return value_string

    def _exception_strings(self, baryCoeff, baryExpected, shape_coeff):
        """Return a list of strings of the actual results that violate the expected results and a boolean that is True if all the results are as expected."""
        exception_strings = []
        rtol = 1e-7  # cchris28: TODO: restore this to 1e-5 before main merge or risk breaking the tests on other platforms
        atol = 1e-20  # cchris28: TODO: restore this to 1e-14 before main merge or risk breaking the tests on other platforms
        for i in range(0, shape_coeff[0]):
            for j in range(1, shape_coeff[1]):
                if not np.isclose(baryCoeff[i, j], baryExpected[i, j], rtol=rtol, atol=atol):
                    exception_strings.append(
                        f"baryCoeff[{i},{j}] = {baryCoeff[i,j]:.16f} != {baryExpected[i,j]:.16f}"
                    )
        return exception_strings

    def _check_barycentric(self, baryCoeff, baryExpected):
        """Check the barycentric correction values against the expected values."""
        # Check the barycentric correction values
        shape_coeff = baryCoeff.shape
        value_string = self._expected_string(baryCoeff, shape_coeff)
        # print(value_string)
        shape_expected = baryExpected.shape
        self.assertEqual(shape_coeff, shape_expected)
        self.assertEqual(len(shape_coeff), 2)
        exception_strings = self._exception_strings(baryCoeff, baryExpected, shape_coeff)
        if len(exception_strings) > 0:
            print(f"Exception strings = {exception_strings}")
            return False
        return True

    def _construct_wcs(
        self,
        img_shape=[256, 256],
        ref_pix=None,
        ref_val=SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs"),
        pixel_scale=0.01 * u.deg / u.pix,
    ):
        """Construct a WCS object for testing."""
        # The reference pixel as used in astropy.wcs.WCS is one based.
        # So the zero based pixel coordinate of the image center is
        #   (image_shape-1)/2
        # whereas the one based pixel coordinate of the image center is
        #   (image_shape-1)/2 + 1 == (image_shape+1)/2 == image_shape//2
        if ref_pix is None:
            ref_pix = [img_shape[0] // 2, img_shape[1] // 2]
        # Define the WCS transformation
        pixel_scale_value = pixel_scale.to(u.deg / u.pix).value
        wcsup = astropy.wcs.WCS(naxis=2)
        wcsup.array_shape = img_shape
        wcsup.wcs.crpix = ref_pix
        wcsup.wcs.cdelt = [-pixel_scale.value, pixel_scale.value]
        wcsup.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcsup.wcs.crval = [ref_val.ra.deg, ref_val.dec.deg]
        return wcsup

    def _synthetic_wcs(self, params):
        """A DRY function that tests the barycentric correction values for a special ra-dec."""
        t = Time(params.times, format="isot", scale="utc")
        visit_times = t.mjd

        # Define the reference pixel and reference value
        # The reference pixel is perpendicular to the tangent plane by definition.
        img_shape = [256, 256]
        ref_pix = [128, 128]
        # Define the celestial coordinates of the tangent point
        ref_val = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs")
        # Define the pixel scale in degrees per pixel
        pixel_scale = 0.01 * u.deg / u.pix

        wcsup = self._construct_wcs(img_shape, ref_pix, ref_val, pixel_scale)

        img_info = kbmod.image_info.ImageInfoSet()
        for i in range(len(params.times)):
            header_info = kbmod.analysis_utils.ImageInfo()
            header_info.width = img_shape[0]
            header_info.height = img_shape[1]
            header_info.ra = params.radeg
            header_info.dec = params.decdeg
            header_info.wcs = wcsup
            header_info.set_epoch(t[i])
            img_info.append(header_info)
        img_info.set_times_mjd(np.array(visit_times))

        run_search = kbmod.run_search.run_search(self.input_parameters)
        baryCoeff = run_search._calc_barycentric_corr(img_info, params.dist)
        return baryCoeff

    def test_image_stack(self):
        # Test the calc_barycentric function of run_search
        run_search = kbmod.run_search.run_search(self.input_parameters)
        self.assertIsNotNone(run_search)
        # Load the PSF.
        kb_interface = kbmod.analysis_utils.Interface()
        default_psf = kbmod.search.psf(run_search.config["psf_val"])

        # Load images to search
        stack, img_info = kb_interface.load_images(
            run_search.config["im_filepath"],
            run_search.config["time_file"],
            run_search.config["psf_file"],
            run_search.config["mjd_lims"],
            default_psf,
            verbose=run_search.config["debug"],
        )
        baryCoeff = run_search._calc_barycentric_corr(img_info, 50.0)
        self.assertIsNotNone(baryCoeff)
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
            [    -1.1841515620267211,     -0.0000003401901527,      0.0000000009598210,     -2.9702163523510885,      0.0000000004351304,     -0.0000003379561716],
            [    -1.9735671031906754,     -0.0000005674411201,      0.0000000015996900,     -4.9503275906180564,      0.0000000007252046,     -0.0000005637178156],
            [  -100.9202381283850798,     -0.0000319903512675,      0.0000000817994955,   -253.2309265796297382,      0.0000000370963011,     -0.0000317998878157],
            [  -102.1013035398333670,     -0.0000324007246175,      0.0000000827567619,   -256.1955780511078160,      0.0000000375305764,     -0.0000322080313739],
            [  -102.8886591105074473,     -0.0000326747631053,      0.0000000833949229,   -258.1719728533235525,      0.0000000378200863,     -0.0000324805833438],
            [  -201.5607485205129024,     -0.0000699606225079,      0.0000001633666874,   -505.9449406539421261,      0.0000000741141381,     -0.0000695800875374],
            [  -202.7383401704241805,     -0.0000704409557553,      0.0000001643210668,   -508.9030324275871067,      0.0000000745474367,     -0.0000700581959558],
            [  -203.5233773925634750,     -0.0000707616325463,      0.0000001649572917,   -510.8750476689951370,      0.0000000748362879,     -0.0000703773895892],
            [  -301.8886731240853010,     -0.0001138896682213,      0.0000002446735039,   -758.0578046608508203,      0.0000001110405992,     -0.0001133195204643],
        ])
        # fmt: on
        self.assertTrue(self._check_barycentric(baryCoeff, baryExpected))

    def test_single_image(self):
        """Verifies that the barycentric corrections for a single image are zeros."""
        run_search = kbmod.run_search.run_search(self.input_parameters)
        self.assertIsNotNone(run_search)
        # Load the PSF.
        kb_interface = kbmod.analysis_utils.Interface()
        default_psf = kbmod.search.psf(run_search.config["psf_val"])
        full_file_path = f"{self.input_parameters['im_filepath']}/000000.fits"
        header_info = kbmod.analysis_utils.ImageInfo()
        header_info.populate_from_fits_file(full_file_path)
        time_obj = header_info.get_epoch(none_if_unset=True)
        time_stamp = time_obj.mjd
        visit_times = [time_stamp]
        img_info = kbmod.image_info.ImageInfoSet()
        img_info.append(header_info)
        img_info.set_times_mjd(np.array(visit_times))
        baryCoeff = run_search._calc_barycentric_corr(img_info, 50.0)
        self.assertIsNotNone(baryCoeff)
        self.assertEqual(baryCoeff.shape, (1, 6))
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
        ])
        # fmt: on
        self.assertTrue(self._check_barycentric(baryCoeff, baryExpected))

    def test_synthetic_pair(self):
        """Use a single wcs at different times"""

        # Define the visit times in MJD roughly 6 months apart
        class Params(object):
            pass

        params = Params()
        params.times = ["2022-09-15T00:00:00.00", "2023-03-15T00:00:00.00"]
        params.radeg = 0.0
        params.decdeg = 90.0
        params.dist = 50.0

        baryCoeff = self._synthetic_wcs(params)
        self.assertIsNotNone(baryCoeff)
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
            [   -26.3706009584761887,      0.0019890565513195,     -0.0000905565708884,    227.2786377276544556,     -0.0001001091165963,      0.0027664477302677],
        ])
        # fmt: on
        self.assertTrue(self._check_barycentric(baryCoeff, baryExpected))

    def test_synthetic_triple(self):
        """Use a single wcs at different times"""

        class Params(object):
            pass

        params = Params()
        params.times = ["2022-12-15T00:00:00.00", "2022-09-15T00:00:00.00", "2023-03-15T00:00:00.00"]
        params.radeg = 0.0
        params.decdeg = 90.0
        params.dist = 10.0

        baryCoeff = self._synthetic_wcs(params)
        self.assertIsNotNone(baryCoeff)
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
            [   591.4066990134609796,     -0.0356047728240740,      0.0011557536335343,   -490.9428016921490325,     -0.0074722715951116,     -0.0453981713040401],
            [   464.6995888871394413,     -0.0279058560645151,      0.0009172030988123,    646.8623689722126073,      0.0097626720908371,     -0.0336200244618067],
        ])
        # fmt: on
        self.assertTrue(self._check_barycentric(baryCoeff, baryExpected))

    def test_single_image_barycentric(self):
        """Test the conversion of a single image with a wcs perspectively projected along a vector from the barycenter and viewed from a geocenter"""

        class Params(object):
            pass

        params = Params()
        params.times = ["2022-12-15T00:00:00.00"]
        params.radeg = 0.0
        params.decdeg = 90.0
        params.dist = 10.0
        params.img_shape = [256, 256]
        params.ref_pix = [127.5, 127.5]
        params.fov = 10.0
        # Define the celestial coordinates of the tangent point
        params.ref_val = SkyCoord(ra=params.radeg * u.deg, dec=params.decdeg * u.deg, frame="icrs")
        # Define the pixel scale in degrees per pixel
        params.pixel_scale = (
            math.atan((math.tan(params.fov * math.pi / 360.0) * 2) / params.img_shape[0])
            * 180.0
            / math.pi
            * u.deg
            / u.pix
        )
        # * u.deg / u.pix

        wcs = self._construct_wcs(params.img_shape, params.ref_pix, params.ref_val, params.pixel_scale)
        assert wcs is not None

        # Convert the image pixels coordinates into celestial coordinates using the wcs.
        xlist, ylist = np.mgrid[0 : params.img_shape[0], 0 : params.img_shape[1]]
        cobs = wcs.pixel_to_world(xlist, ylist)
        projection_vector = cobs[int(params.ref_pix[0]), int(params.ref_pix[1])].cartesian
        cobsproj = cobs.cartesian * ((params.dist * astropy.units.au) / cobs.cartesian.dot(projection_vector))
        assert cobs is not None

    def test_least_squares_fit_parameters(self):
        """Test the least squares fit parameters
        Construct a tangent plane wcs at ninety degrees from the ICRS equator with
        a pixel scale of 2 arcsec/pixel and a reference pixel at the center of the image.
        Choose two observer positions at 1 au from the barycenter with 0 degrees declination
        and 0 and 180 degrees right ascension, respectively using the same WCS. Construct
        barycentric correction factors for the two observer positions for an object at 50.0 au
        along the line of site from the barycenter. The barycentric correction factors should
        by symmetric for the two observers.
        """
        # Test the calc_barycentric function of run_search
        input_parameters = {
            # Required but unused by this test
            "im_filepath": "../data/demo",
        }

        run_search = kbmod.run_search.run_search(input_parameters)
        self.assertIsNotNone(run_search)

        # Create a WCS with a reference pixel at the center of the image
        # use an odd number of pixels so that the center of the image is
        # also the center of a pixel. This aligns visually.
        pixel_fov = 0.2 * u.arcsec / u.pix * (4096 / 2)  # lsst size
        image_ref_pix = 0  # 0 or more works, zero based
        barycenter_object_dist = 5.0 * u.au
        barycenter_observer_dist = 1.0 * u.au

        image_size = image_ref_pix * 2 + 1
        img_shape = [image_size, image_size]
        # astropy.wcs uses one based pixel coordinates for the reference pixel.
        ref_pix = [image_ref_pix + 1, image_ref_pix + 1]
        # Define the celestial coordinates of the tangent point
        # Avoid the pole though that really is silly. It is enough to want to use quaternions.
        # ra needs to be 90 for this test but dec could be anything away from the poles.
        ref_val = SkyCoord(ra=90 * u.deg, dec=0 * u.deg, frame="icrs")
        # Define the pixel scale in degrees per pixel
        pixel_scale = pixel_fov / image_size
        wcsup = self._construct_wcs(img_shape, ref_pix, ref_val, pixel_scale)
        self.assertIsNotNone(wcsup)
        # Three observers. One at the barycenter and two at 1 au from the barycenter
        # with 0 and 180 degrees right ascension, respectively, so that with a ref_val.ra if 90 degrees
        # they make an isoceles triangle with the barycenter splitting the base.
        barycenter_observer_dist_value = barycenter_observer_dist.to(u.au).value
        observer_position_list = [
            SkyCoord(x=0, y=0, z=0, unit="au", representation_type="cartesian").cartesian,
            SkyCoord(
                x=barycenter_observer_dist_value, y=0, z=0, unit="au", representation_type="cartesian"
            ).cartesian,
            SkyCoord(
                x=-barycenter_observer_dist_value, y=0, z=0, unit="au", representation_type="cartesian"
            ).cartesian,
        ]
        observer_wcs_list = [wcsup, wcsup, wcsup]
        xlist, ylist = np.mgrid[0:image_size, 0:image_size]
        reference_positions = wcsup.pixel_to_world(xlist, ylist).cartesian * barycenter_object_dist
        run_search.wcslist = observer_wcs_list
        run_search.obs_pos_list = observer_position_list
        # TODO: making this a SkyCoord is an extra step but it is what the code expects.
        # The least squares code seems to need the data flattened.
        run_search.cbary = SkyCoord(reference_positions, representation_type="cartesian").flatten()
        xlist = xlist.flatten()
        ylist = ylist.flatten()
        cbarycorr = kbmod.run_search.run_search._calculate_barycoeff_list(
            xlist, ylist, observer_wcs_list, run_search.cbary, observer_position_list
        )
        self.assertIsNotNone(cbarycorr)
        # given the symmetry of the problem, the barycentric correction factors should be symmetric
        # specifically, the only non-zero values should be the dx and dy values and they should sum
        # to zero. There likely will be some small numerical error.
        self.assertAlmostEqual(cbarycorr[1, 0], -cbarycorr[2, 0], places=10)
        self.assertTrue(np.allclose(0, cbarycorr[:, [1, 2, 4, 5]], atol=1e-08))


if __name__ == "__main__":
    unittest.main()
