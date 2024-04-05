import unittest

import numpy as np
import numpy.testing as npt
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.wcs import WCS

from kbmod.reprojection_utils import correct_parallax, fit_barycentric_wcs


class test_reprojection_utils(unittest.TestCase):
    def test_parallax_equinox(self):
        icrs_ra1 = 88.74513571
        icrs_dec1 = 23.43426475
        time1 = Time("2023-03-20T16:00:00", format="isot", scale="utc")

        icrs_ra2 = 91.24261107
        icrs_dec2 = 23.43437467
        time2 = Time("2023-09-24T04:00:00", format="isot", scale="utc")

        sc1 = SkyCoord(ra=icrs_ra1, dec=icrs_dec1, unit="deg")
        sc2 = SkyCoord(ra=icrs_ra2, dec=icrs_dec2, unit="deg")

        with solar_system_ephemeris.set("de432s"):
            loc = EarthLocation.of_site("ctio")

        corrected_coord1 = correct_parallax(
            coord=sc1,
            obstime=time1,
            point_on_earth=loc,
            guess_distance=50.0,
        )

        expected_ra = 90.0
        expected_dec = 23.43952556

        npt.assert_almost_equal(corrected_coord1.ra.value, expected_ra)
        npt.assert_almost_equal(corrected_coord1.dec.value, expected_dec)

        corrected_coord2 = correct_parallax(
            coord=sc2,
            obstime=time2,
            point_on_earth=loc,
            guess_distance=50.0,
        )

        npt.assert_almost_equal(corrected_coord2.ra.value, expected_ra)
        npt.assert_almost_equal(corrected_coord2.dec.value, expected_dec)

    def test_fit_barycentric_wcs(self):
        nx = 2046
        ny = 4094
        test_wcs = WCS(naxis=2)
        test_wcs.pixel_shape = (ny, nx)
        test_wcs.wcs.crpix = [nx / 2, ny / 2]
        test_wcs.wcs.cdelt = np.array([-0.000055555555556, 0.000055555555556])
        test_wcs.wcs.crval = [346.9681342111, -6.482196848597]
        test_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

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

        time = "2021-08-24T20:59:06"
        site = "ctio"
        loc = EarthLocation.of_site(site)
        distance = 41.1592725489203

        corrected_wcs = fit_barycentric_wcs(
            test_wcs,
            nx,
            ny,
            distance,
            time,
            loc,
        )

        corrected_ra, corrected_dec = corrected_wcs.all_pix2world(x_points, y_points, 0)
        corrected_sc = SkyCoord(corrected_ra, corrected_dec, unit="deg")
        seps = expected_sc.separation(corrected_sc).arcsecond

        # assert we have sub-milliarcsecond precision
        assert np.all(seps < 0.001)
