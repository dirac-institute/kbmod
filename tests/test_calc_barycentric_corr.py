import unittest

import astropy.wcs
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import math

import kbmod
import kbmod.analysis_utils
import kbmod.search


class test_calc_barycentric_corr(unittest.TestCase):
    def setUp(self):
        # Define the path for the data.
        # TODO: this should be ../data/demo but vscode does not get the message all the time.
        # from time import sleep
        # sleep(300)
        im_filepath = "../data/demo"
        # HACK: workaround for vscode not honoring the cwd specification.
        from os.path import isdir

        if not isdir(im_filepath):
            im_filepath = "data/demo"
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
        check_ok = True
        for i in range(0, shape_coeff[0]):
            for j in range(1, shape_coeff[1]):
                if not np.isclose(baryCoeff[i, j], baryExpected[i, j], rtol=1e-5, atol=1e-14):
                    exception_strings.append(
                        f"baryCoeff[{i},{j}] = {baryCoeff[i,j]:.16f} != {baryExpected[i,j]:.16f}"
                    )
                    check_ok = False
        return exception_strings, check_ok

    def _check_barycentric(self, baryCoeff, baryExpected):
        """Check the barycentric correction values against the expected values."""
        # Check the barycentric correction values
        shape_coeff = baryCoeff.shape
        value_string = self._expected_string(baryCoeff, shape_coeff)
        # print(value_string)
        shape_expected = baryExpected.shape
        assert shape_coeff == shape_expected
        assert len(shape_coeff) == 2
        exception_strings, check_ok = self._exception_strings(baryCoeff, baryExpected, shape_coeff)
        if len(exception_strings) > 0:
            print(f"Exception strings = {exception_strings}")
        return check_ok

    def _construct_wcs(
        self,
        img_shape=[256, 256],
        ref_pix=None,
        ref_val=SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs"),
        pixel_scale=0.01 * u.deg / u.pix,
    ):
        """Construct a WCS object for testing."""
        if ref_pix is None:
            ref_pix = [img_shape[0] // 2, img_shape[1] // 2]
        # Define the WCS transformation
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
        assert run_search is not None
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
        assert baryCoeff is not None
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
        assert self._check_barycentric(baryCoeff, baryExpected)

    def test_single_image(self):
        """Verifies that the barycentric corrections for a single image are zeros."""
        run_search = kbmod.run_search.run_search(self.input_parameters)
        assert run_search is not None
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
        assert baryCoeff is not None
        assert baryCoeff.shape == (1, 6)
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
        ])
        # fmt: on
        assert self._check_barycentric(baryCoeff, baryExpected)

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
        assert baryCoeff is not None
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
            [   -26.3706009584761887,      0.0019890565513195,     -0.0000905565708884,    227.2786377276544556,     -0.0001001091165963,      0.0027664477302677],
        ])
        # fmt: on
        assert self._check_barycentric(baryCoeff, baryExpected)

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
        assert baryCoeff is not None
        # fmt: off
        baryExpected = np.array([
            #       dx,                      dxdx,                    dxdy,                    dy,                      dydx,                    dydy,
            [     0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000,      0.0000000000000000],
            [   591.4066990134609796,     -0.0356047728240740,      0.0011557536335343,   -490.9428016921490325,     -0.0074722715951116,     -0.0453981713040401],
            [   464.6995888871394413,     -0.0279058560645151,      0.0009172030988123,    646.8623689722126073,      0.0097626720908371,     -0.0336200244618067],
        ])
        # fmt: on
        assert self._check_barycentric(baryCoeff, baryExpected)


if __name__ == "__main__":
    unittest.main()
