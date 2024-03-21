import unittest

import numpy.testing as npt
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time

from kbmod.reprojection_utils import correct_parallax


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

        with solar_system_ephemeris.set('de432s'):
            loc = EarthLocation.of_site('ctio')

        corrected_coord1 = correct_parallax(
            coord=sc1,
            obstime=time1,
            point_on_earth=loc,
            guess_distance=50.,
        )

        expected_ra = 90.
        expected_dec = 23.43952556

        npt.assert_almost_equal(corrected_coord1.ra.value, expected_ra)
        npt.assert_almost_equal(corrected_coord1.dec.value, expected_dec)

        corrected_coord2 = correct_parallax(
            coord=sc2,
            obstime=time2,
            point_on_earth=loc,
            guess_distance=50.,
        )

        npt.assert_almost_equal(corrected_coord2.ra.value, expected_ra)
        npt.assert_almost_equal(corrected_coord2.dec.value, expected_dec)