from astropy.time import Time
import astropy.units as u
from os import path
import numpy as np
import tempfile
import unittest

from kbmod.region_search.pointing_table import PointingTable
from utils.utils_for_tests import get_absolute_data_path


class test_pointings_table(unittest.TestCase):
    def test_check_and_rename_column(self):
        d = {
            "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
            "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
        }
        data = PointingTable.from_dict(d)

        # The column is there.
        self.assertTrue(data._check_and_rename_column("ra", [], True))
        self.assertTrue(data._check_and_rename_column("ra", [], False))

        # The column is not there, but can be (and is) renamed.
        self.assertFalse("dec" in data.pointings.columns)
        self.assertTrue(data._check_and_rename_column("dec", ["Dec", "DEC", "declin"], True))
        self.assertTrue("dec" in data.pointings.columns)

        # A column is missing without a valid replacement.
        self.assertFalse(data._check_and_rename_column("time", ["mjd", "MJD", "obstime"], False))
        with self.assertRaises(ValueError):
            data._check_and_rename_column("time", ["mjd", "MJD", "obstime"], True)

    def test_validate_and_standardize(self):
        d = {
            "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
            "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
            "MJD": [0.0, 1.0, 2.0, 3.0, 4.0],
            "brightness": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
        data = PointingTable.from_dict(d)

        self.assertEqual(len(data.pointings.columns), 4)
        self.assertTrue("ra" in data.pointings.columns)
        self.assertFalse("dec" in data.pointings.columns)
        self.assertFalse("obstime" in data.pointings.columns)
        self.assertFalse("flux" in data.pointings.columns)

        data._validate_and_standardize()
        self.assertEqual(len(data.pointings.columns), 4)
        self.assertTrue("ra" in data.pointings.columns)
        self.assertTrue("dec" in data.pointings.columns)
        self.assertTrue("obstime" in data.pointings.columns)
        self.assertFalse("flux" in data.pointings.columns)

        data._validate_and_standardize({"flux": ["brightness"]})
        self.assertEqual(len(data.pointings.columns), 4)
        self.assertTrue("ra" in data.pointings.columns)
        self.assertTrue("dec" in data.pointings.columns)
        self.assertTrue("obstime" in data.pointings.columns)
        self.assertTrue("flux" in data.pointings.columns)

    def test_from_csv(self):
        filename = path.join(get_absolute_data_path(), "test_pointings.csv")
        data = PointingTable.from_csv(filename)
        self.assertEqual(len(data.pointings), 5)
        self.assertEqual(len(data.pointings.columns), 5)

    def test_to_csv(self):
        d = {
            "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
            "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
            "MJD": [0.0, 1.0, 2.0, 3.0, 4.0],
            "flux": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
        data = PointingTable.from_dict(d)

        with tempfile.TemporaryDirectory() as dir_name:
            filename = path.join(dir_name, "test.csv")
            data.to_csv(filename)

            # Check that we can reload it.
            data2 = PointingTable.from_csv(filename)
            self.assertEqual(len(data.pointings), 5)
            self.assertEqual(len(data.pointings.columns), 4)

    def test_append_sun_pos(self):
        d = {
            "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
            "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
            "obstime": [60253.0 + i / 24.0 for i in range(5)],
        }
        data = PointingTable.from_dict(d)
        self.assertFalse("sun_pos" in data.pointings.columns)
        self.assertFalse("sun_vec" in data.pointings.columns)

        # Check that the data is corrected.
        data.append_sun_pos()
        self.assertTrue("sun_pos" in data.pointings.columns)
        self.assertTrue("sun_vec" in data.pointings.columns)
        self.assertEqual(len(data.pointings["sun_pos"]), 5)
        self.assertEqual(len(data.pointings["sun_vec"]), 5)

        # Check that the sun's distance is reasonable and consistent between the
        # angular and cartesian representations.
        for i in range(5):
            self.assertLess(data.pointings["sun_pos"][i].distance, 1.1 * u.AU)
            self.assertGreater(data.pointings["sun_pos"][i].distance, 0.9 * u.AU)
            vec_dist = np.linalg.norm(data.pointings["sun_vec"][i])
            self.assertTrue(np.isclose(data.pointings["sun_pos"][i].distance.value, vec_dist))

    def test_append_unit_vector(self):
        d = {
            "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
            "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        }
        data = PointingTable.from_dict(d)
        self.assertFalse("unit_vec" in data.pointings.columns)

        data.append_unit_vector()
        self.assertTrue("unit_vec" in data.pointings.columns)
        self.assertTrue(np.allclose(data.pointings["unit_vec"][:, 0], [1.0, 0.0, 0.707106781, 0.0, 0.0]))
        self.assertTrue(
            np.allclose(data.pointings["unit_vec"][:, 1], [0.0, 0.0, 0.707106781, 0.707106781, -1.0])
        )
        self.assertTrue(np.allclose(data.pointings["unit_vec"][:, 2], [0.0, 1.0, 0.0, 0.707106781, 0.0]))

    def test_angular_dist_3d_heliocentric(self):
        # The first observation is effectively looking at the sun and the second is
        # looking 1 degree away.
        d = {
            "ra": [219.63062629578198, 219.63062629578198],
            "dec": [-15.455316915908792, -16.455316915908792],
            "obstime": [60253.1, 60253.1],
        }
        data = PointingTable.from_dict(d)

        # Check the pointings compared to the position of the sun.
        ang_dist = data.angular_dist_3d_heliocentric([0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(ang_dist, [0.0, 1.0], atol=1e-5))

        # Check an object that is 1 AU from the sun along the x-axis
        ang_dist = data.angular_dist_3d_heliocentric([1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(ang_dist, [69.587114, 69.283768], atol=1e-5))

        # Check an object in a known location in geocentric space [0.5, 0.5, 0.0] when looking
        # out at RA=0.0 and dec=0.0
        d2 = {"ra": [0.0], "dec": [0.0], "obstime": [60253.1]}
        data2 = PointingTable.from_dict(d2)
        ang_dist = data2.angular_dist_3d_heliocentric(
            [1.2361460125166166, 1.1096560270277159, 0.2642697128572278]
        )
        self.assertTrue(np.allclose(ang_dist, 45.0, atol=1e-5))

    def test_angular_dist_3d_heliocentric(self):
        # The first observation is effectively looking at the sun.
        d = {
            "obsid": [1, 2, 3, 4, 5, 6],
            "ra": [219.63063, 219.63063, 219.63063, 219.63063, 25.51, 356.24],
            "dec": [-15.45532, -16.45532, -15.7, -15.45532, 15.45532, -1.6305],
            "obstime": [60253.1, 60253.1, 60253.1, 60353.5, 60253.1, 60253.1],
        }
        data = PointingTable.from_dict(d)

        # Check the pointings compared to the position of the sun.
        match_table = data.search_heliocentric_pointing(0.0, 0.0, 0.0, 0.9)
        self.assertEqual(len(match_table), 2)
        self.assertTrue(np.allclose(match_table["obsid"], [1, 3]))

        # Check the pointings 10 AU out from the sun.
        match_table = data.search_heliocentric_pointing(0.0, 0.0, 10.0, 0.9)
        self.assertEqual(len(match_table), 1)
        self.assertTrue(np.allclose(match_table["obsid"], [6]))
