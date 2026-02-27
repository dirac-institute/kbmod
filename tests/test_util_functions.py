from astropy.table import Table
import numpy as np
import unittest
from uuid import uuid4

from pathlib import Path

from kbmod.core.image_stack_py import LayeredImagePy
from kbmod.core.psf import PSF
from kbmod.results import Results
from kbmod.search import Trajectory
from kbmod.util_functions import (
    get_matched_obstimes,
    load_deccam_layered_image,
    make_manual_tracklets,
    mjd_to_day,
    standardize_ephemeris_coordinates,
    standardize_ephemeris_time,
    unravel_results,
)


class test_util_functions(unittest.TestCase):
    def setUp(self):
        trj_list = []
        self.num_images = 10
        self.num_trjs = 11

        for i in range(self.num_trjs):
            trj = Trajectory(
                x=i,
                y=i + 0,
                vx=i - 2.0,
                vy=i + 5.0,
                flux=5.0 * i + 1.0,  # Flux must be > 1
                lh=100.0 + i,
                obs_count=self.num_images,
            )
            trj_list.append(trj)

        res = Results.from_trajectories(trj_list)

        # create an "image collection" with all required fields
        ic = Table()
        ic["mjd_mid"] = [60000 + i for i in range(self.num_images)]
        ic["band"] = ["g" for _ in range(self.num_images)]
        ic["zeroPoint"] = [31.4 for _ in range(self.num_images)]

        res.table.meta["mjd_mid"] = ic["mjd_mid"]
        obs_count = [self.num_images for _ in range(self.num_trjs)]
        res.table["obs_count"] = obs_count
        res.table["img_ra"] = [
            np.array([j + (i * 0.1) for j in range(self.num_images)]) for i in range(self.num_trjs)
        ]
        res.table["img_dec"] = [
            np.array([j + (i * 0.1) for j in range(self.num_images)]) for i in range(self.num_trjs)
        ]

        self.ic = ic
        self.res = res

    def test_get_matched_obstimes(self):
        obstimes = [1.0, 2.0, 3.0, 4.0, 6.0, 7.5, 9.0, 10.1]
        query_times = [-1.0, 0.999999, 1.001, 1.1, 1.999, 6.001, 7.499, 7.5, 10.099999, 10.10001, 20.0]
        matched_inds = get_matched_obstimes(obstimes, query_times, threshold=0.01)
        expected = [-1, 0, 0, -1, 1, 4, 5, 5, 7, 7, -1]
        self.assertTrue(np.array_equal(matched_inds, expected))

    def test_mjd_to_day(self):
        self.assertEqual(mjd_to_day(60481.04237269), "2024-06-20")
        self.assertEqual(mjd_to_day(60500), "2024-07-09")
        self.assertEqual(mjd_to_day(58100.5), "2017-12-13")

    def test_load_deccam_layered_image(self):
        base_path = Path(__file__).parent.parent
        img_path = base_path / "data" / "demo_image.fits"
        img = load_deccam_layered_image(img_path, PSF.make_gaussian_kernel(1.0))

        self.assertTrue(isinstance(img, LayeredImagePy))
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)

    def test_unravel_results(self):
        df = unravel_results(self.res, self.ic)
        self.assertEqual(len(df), (self.num_images * self.num_trjs))

        obs_count = self.res.table["obs_count"]
        obs_count[int(self.num_images / 2)] = self.num_images - 1
        valids = [[True] * self.num_images for _ in range(self.num_trjs)]
        valids[int(self.num_images / 2)][-1] = False  # make one observation invalid
        res2 = self.res
        res2.table["obs_valid"] = valids
        res2.table["obs_count"] = obs_count

        df2 = unravel_results(res2, self.ic)
        self.assertEqual(len(df2), (self.num_images * self.num_trjs) - 1)

        df2 = unravel_results(self.res, self.ic, first_and_last=True)
        self.assertEqual(len(df2), self.num_trjs * 2)

    def test_make_manual_tracklets(self):
        df = unravel_results(self.res, self.ic)
        tracklets, trk2det = make_manual_tracklets(df)

        self.assertEqual(len(tracklets), (self.num_trjs * (self.num_images - 1)))
        self.assertEqual(len(trk2det), self.num_trjs * ((self.num_images - 1) * 2))

        for trk_idx in range(self.num_trjs):
            for img_idx in range(self.num_images - 1):
                tracklet = tracklets.iloc[trk_idx * (self.num_images - 1) + img_idx]
                det1 = tracklet[["#Image1", "RA1", "Dec1"]]
                det2 = tracklet[["Image2", "RA2", "Dec2"]]

                o_idx = trk_idx * (self.num_images) + img_idx
                o_det1 = df.iloc[o_idx]
                o_det2 = df.iloc[o_idx + 1]

                self.assertEqual(det1["#Image1"], o_det1["mjd"])
                self.assertEqual(det1["RA1"], o_det1["ra"])
                self.assertEqual(det1["Dec1"], o_det1["dec"])

                self.assertEqual(det2["Image2"], o_det2["mjd"])
                self.assertEqual(det2["RA2"], o_det2["ra"])
                self.assertEqual(det2["Dec2"], o_det2["dec"])

    def test_make_manual_tracklets_without_uuid(self):
        test_res = self.res
        test_res.table.remove_column("uuid")
        df = unravel_results(test_res, self.ic)
        self.assertRaises(ValueError, make_manual_tracklets, df)

    # --- standardize_ephemeris_time tests ---

    def test_time_passthrough_if_mjd_mid_exists(self):
        t = Table({"mjd_mid": [60000.0, 60001.0], "other": [1, 2]})
        result = standardize_ephemeris_time(t)
        np.testing.assert_array_equal(result["mjd_mid"], [60000.0, 60001.0])

    def test_time_obs_time_jpl(self):
        t = Table({"obs-time": ["2023-02-25T00:00:00", "2023-02-26T12:00:00"]})
        result = standardize_ephemeris_time(t)
        self.assertIn("mjd_mid", result.colnames)
        self.assertAlmostEqual(result["mjd_mid"][0], 60000.0, places=3)

    def test_time_ref_epoch_float(self):
        t = Table({"ref_epoch": ["60000.0", "60001.5"]})
        result = standardize_ephemeris_time(t)
        np.testing.assert_array_almost_equal(result["mjd_mid"], [60000.0, 60001.5])

    def test_time_ref_epoch_datetime_fallback(self):
        t = Table({"ref_epoch": ["2023-02-25T00:00:00", "2023-02-26T12:00:00"]})
        result = standardize_ephemeris_time(t)
        self.assertAlmostEqual(result["mjd_mid"][0], 60000.0, places=3)

    def test_time_explicit_column(self):
        t = Table({"my_time": ["60100.0", "60200.0"], "obs-time": ["2020-01-01", "2020-01-02"]})
        result = standardize_ephemeris_time(t, column="my_time")
        np.testing.assert_array_almost_equal(result["mjd_mid"], [60100.0, 60200.0])

    def test_time_raises_on_missing_columns(self):
        t = Table({"foo": [1, 2]})
        with self.assertRaises(ValueError):
            standardize_ephemeris_time(t)

    # --- standardize_ephemeris_coordinates tests ---

    def test_coords_passthrough_if_ra_dec_exist(self):
        t = Table({"ra": [10.0, 20.0], "dec": [-5.0, 5.0]})
        result = standardize_ephemeris_coordinates(t)
        np.testing.assert_array_equal(result["ra"], [10.0, 20.0])
        np.testing.assert_array_equal(result["dec"], [-5.0, 5.0])

    def test_coords_rename_RA_Dec(self):
        t = Table({"RA": [10.0, 20.0], "Dec": [-5.0, 5.0]})
        result = standardize_ephemeris_coordinates(t)
        self.assertIn("ra", result.colnames)
        self.assertIn("dec", result.colnames)
        np.testing.assert_array_equal(result["ra"], [10.0, 20.0])
        np.testing.assert_array_equal(result["dec"], [-5.0, 5.0])

    def test_coords_jpl_format(self):
        t = Table(
            {
                "Astrometric RA (hh:mm:ss)": ["12 30 00.0", "06 15 00.0"],
                "Astrometric Dec (dd mm'ss\")": ["+45 00 00.0", "-17 23' 27.0\""],
            }
        )
        result = standardize_ephemeris_coordinates(t)
        self.assertAlmostEqual(result["ra"][0], 187.5, places=3)
        self.assertAlmostEqual(result["dec"][0], 45.0, places=3)

    def test_coords_skybot_format(self):
        t = Table(
            {
                "RA (hms)": ["12 30 00.0", "06 15 00.0"],
                "DEC (dms)": ["+45 00 00.0", "-30 00 00.0"],
            }
        )
        result = standardize_ephemeris_coordinates(t)
        self.assertAlmostEqual(result["ra"][0], 187.5, places=3)
        self.assertAlmostEqual(result["dec"][0], 45.0, places=3)
        self.assertAlmostEqual(result["dec"][1], -30.0, places=3)

    def test_coords_explicit_columns(self):
        t = Table({"my_ra": ["12 00 00.0"], "my_dec": ["+45 00 00.0"], "RA": [999.0]})
        result = standardize_ephemeris_coordinates(t, ra_column="my_ra", dec_column="my_dec")
        self.assertAlmostEqual(result["ra"][0], 180.0, places=3)
        self.assertAlmostEqual(result["dec"][0], 45.0, places=3)

    def test_coords_raises_on_missing_ra(self):
        t = Table({"dec": [1.0], "foo": [2]})
        with self.assertRaises(ValueError):
            standardize_ephemeris_coordinates(t)

    def test_coords_raises_on_missing_dec(self):
        t = Table({"ra": [1.0], "foo": [2]})
        with self.assertRaises(ValueError):
            standardize_ephemeris_coordinates(t)

    def test_coords_negative_dec(self):
        t = Table({"RA": [10.0], "Astrometric Dec (dd mm'ss\")": ["-17 23' 27.0\""]})
        result = standardize_ephemeris_coordinates(t)
        self.assertAlmostEqual(result["dec"][0], -17.3908, places=3)


if __name__ == "__main__":
    unittest.main()
