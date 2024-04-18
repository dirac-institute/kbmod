import unittest

import numpy as np
import numpy.testing as npt
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.wcs import WCS

from kbmod.reprojection_utils import correct_parallax, invert_correct_parallax, fit_barycentric_wcs


class test_reprojection_utils(unittest.TestCase):
    def setUp(self):
        self.nx = 2046
        self.ny = 4094
        self.test_wcs = WCS(naxis=2)
        self.test_wcs.pixel_shape = (self.ny, self.nx)
        self.test_wcs.wcs.crpix = [self.nx / 2, self.ny / 2]
        self.test_wcs.wcs.cdelt = np.array([-0.000055555555556, 0.000055555555556])
        self.test_wcs.wcs.crval = [346.9681342111, -6.482196848597]
        self.test_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        self.time = "2021-08-24T20:59:06"
        self.site = "ctio"
        self.loc = EarthLocation.of_site(self.site)
        self.distance = 41.1592725489203

        self.icrs_ra1 = 88.74513571
        self.icrs_dec1 = 23.43426475
        self.icrs_time1 = Time("2023-03-20T16:00:00", format="isot", scale="utc")

        self.icrs_ra2 = 91.24261107
        self.icrs_dec2 = 23.43437467
        self.icrs_time2 = Time("2023-09-24T04:00:00", format="isot", scale="utc")

        self.sc1 = SkyCoord(ra=self.icrs_ra1, dec=self.icrs_dec1, unit="deg")
        self.sc2 = SkyCoord(ra=self.icrs_ra2, dec=self.icrs_dec2, unit="deg")

        with solar_system_ephemeris.set("de432s"):
            self.eq_loc = EarthLocation.of_site("ctio")

    def test_parallax_equinox(self):
        corrected_coord1, _ = correct_parallax(
            coord=self.sc1,
            obstime=self.icrs_time1,
            point_on_earth=self.eq_loc,
            heliocentric_distance=50.0,
        )

        expected_ra = 90.0
        expected_dec = 23.43952556

        npt.assert_almost_equal(corrected_coord1.ra.value, expected_ra)
        npt.assert_almost_equal(corrected_coord1.dec.value, expected_dec)

        corrected_coord2, _ = correct_parallax(
            coord=self.sc2,
            obstime=self.icrs_time2,
            point_on_earth=self.eq_loc,
            heliocentric_distance=50.0,
        )

        npt.assert_almost_equal(corrected_coord2.ra.value, expected_ra)
        npt.assert_almost_equal(corrected_coord2.dec.value, expected_dec)

        assert type(corrected_coord1) is SkyCoord
        assert type(corrected_coord2) is SkyCoord

    def test_invert_correct_parallax(self):
        corrected_coord1, geo_dist1 = correct_parallax(
            coord=self.sc1,
            obstime=self.icrs_time1,
            point_on_earth=self.eq_loc,
            heliocentric_distance=50.0,
        )

        fresh_sc1 = SkyCoord(ra=corrected_coord1.ra.degree, dec=corrected_coord1.dec.degree, unit="deg")

        uncorrected_coord1 = invert_correct_parallax(
            coord=fresh_sc1,
            obstime=self.icrs_time1,
            point_on_earth=self.eq_loc,
            geocentric_distance=geo_dist1,
            heliocentric_distance=50.0,
        )

        assert self.sc1.separation(uncorrected_coord1).arcsecond < 0.001

        corrected_coord2, geo_dist2 = correct_parallax(
            coord=self.sc2,
            obstime=self.icrs_time2,
            point_on_earth=self.eq_loc,
            heliocentric_distance=50.0,
        )

        fresh_sc2 = SkyCoord(ra=corrected_coord2.ra.degree, dec=corrected_coord2.dec.degree, unit="deg")

        uncorrected_coord2 = invert_correct_parallax(
            coord=fresh_sc2,
            obstime=self.icrs_time2,
            point_on_earth=self.eq_loc,
            geocentric_distance=geo_dist2,
            heliocentric_distance=50.0,
        )

        assert self.sc2.separation(uncorrected_coord2).arcsecond < 0.001

    def test_fit_barycentric_wcs(self):
        x_points = np.array([247, 1252, 1052, 980, 420, 1954, 730, 1409, 1491, 803])
        y_points = np.array([1530, 713, 3414, 3955, 1975, 123, 1456, 2008, 1413, 1756])

        expected_ra = np.array(
            [
                346.69225567,
                346.63734563,
                346.64836252,
                346.65231188,
                346.68282256,
                346.59898412,
                346.66587788,
                346.62881986,
                346.6243199,
                346.66190162,
            ]
        )

        expected_dec = np.array(
            [
                -6.62151717,
                -6.66580019,
                -6.51929901,
                -6.48995635,
                -6.5973741,
                -6.6977762,
                -6.62551611,
                -6.59555108,
                -6.62782211,
                -6.60924105,
            ]
        )

        expected_sc = SkyCoord(ra=expected_ra, dec=expected_dec, unit="deg")

        corrected_wcs, geo_dist = fit_barycentric_wcs(
            self.test_wcs,
            self.nx,
            self.ny,
            self.distance,
            self.time,
            self.loc,
        )

        corrected_ra, corrected_dec = corrected_wcs.all_pix2world(x_points, y_points, 0)
        corrected_sc = SkyCoord(corrected_ra, corrected_dec, unit="deg")
        seps = expected_sc.separation(corrected_sc).arcsecond

        # assert we have sub-milliarcsecond precision
        assert np.all(seps < 0.001)
        assert corrected_wcs.array_shape == (self.ny, self.nx)
        npt.assert_almost_equal(geo_dist, 40.18622, decimal=4)

    def test_fit_barycentric_wcs_consistency(self):
        corrected_wcs, geo_dist = fit_barycentric_wcs(
            self.test_wcs, self.nx, self.ny, self.distance, self.time, self.loc, seed=24601
        )

        # crval consistency
        npt.assert_almost_equal(corrected_wcs.wcs.crval[0], 346.6498731934591)
        npt.assert_almost_equal(corrected_wcs.wcs.crval[1], -6.593449653602658)

        # crpix consistency
        npt.assert_almost_equal(corrected_wcs.wcs.crpix[0], 1024.4630013095195)
        npt.assert_almost_equal(corrected_wcs.wcs.crpix[1], 2047.9912979360922)

        # cd consistency
        npt.assert_almost_equal(corrected_wcs.wcs.cd[0][0], -5.424296904025753e-05)
        npt.assert_almost_equal(corrected_wcs.wcs.cd[0][1], 3.459611876675614e-08)
        npt.assert_almost_equal(corrected_wcs.wcs.cd[1][0], 3.401472764249802e-08)
        npt.assert_almost_equal(corrected_wcs.wcs.cd[1][1], 5.4242245855217796e-05)

        npt.assert_almost_equal(geo_dist, 40.18622524245729)
