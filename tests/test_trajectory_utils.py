import numpy as np
import unittest

from astropy.wcs import WCS

from kbmod.trajectory_utils import *
from kbmod.search import *


class test_trajectory_utils(unittest.TestCase):
    def test_predict_pixel_locations(self):
        x0 = [0.0, 1.0, 2.0]
        vx = [1.0, 0.0, -1.0]
        times = [0, 1, 2, 3]

        # When centering and using an integer, the last point for x0=2.0 vx=-1.0
        # is at -0.5 which is rounded to 0.0 because we are using int truncation
        # instead of an explicit floor for consistency.
        pos = predict_pixel_locations(times, x0, vx, centered=True, as_int=True)
        expected = np.array([[0, 1, 2, 3], [1, 1, 1, 1], [2, 1, 0, 0]])
        self.assertTrue(np.array_equal(pos, expected))

        pos = predict_pixel_locations(times, x0, vx, centered=False, as_int=True)
        expected = np.array([[0, 1, 2, 3], [1, 1, 1, 1], [2, 1, 0, -1]])
        self.assertTrue(np.array_equal(pos, expected))

        pos = predict_pixel_locations(times, x0, vx, centered=True, as_int=False)
        expected = np.array([[0.5, 1.5, 2.5, 3.5], [1.5, 1.5, 1.5, 1.5], [2.5, 1.5, 0.5, -0.5]])
        self.assertTrue(np.array_equal(pos, expected))

        pos = predict_pixel_locations(times, x0, vx, centered=False, as_int=False)
        expected = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 0.0, -1.0]])
        self.assertTrue(np.array_equal(pos, expected))

    def test_predict_skypos(self):
        # Create a fake WCS with a known pointing.
        my_wcs = WCS(naxis=2)
        my_wcs.wcs.crpix = [10.0, 10.0]  # Reference point on the image (1-indexed)
        my_wcs.wcs.crval = [45.0, -15.0]  # Reference pointing on the sky
        my_wcs.wcs.cdelt = [0.1, 0.1]  # Pixel step size
        my_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

        trj = make_trajectory_from_ra_dec(45.0, -15.0, 1.0, 0.5)
        self.assertAlmostEqual(trj.x, 9.0)
        self.assertAlmostEqual(trj.y, 9.0)
        self.assertAlmostEqual(trj.vx, 9.9190138, delta=1e-6)
        self.assertAlmostEqual(trj.vy, 4.97896903, delta=1e-6)

    def test_predict_skypos(self):
        # Create a fake WCS with a known pointing.
        my_wcs = WCS(naxis=2)
        my_wcs.wcs.crpix = [10.0, 10.0]  # Reference point on the image (1-indexed)
        my_wcs.wcs.crval = [45.0, -15.0]  # Reference pointing on the sky
        my_wcs.wcs.cdelt = [0.1, 0.1]  # Pixel step size
        my_wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]

        # Confirm that the wcs produces the correct prediction (using zero indexed pixel).
        my_sky = my_wcs.pixel_to_world(9.0, 9.0)
        self.assertAlmostEqual(my_sky.ra.deg, 45.0)
        self.assertAlmostEqual(my_sky.dec.deg, -15.0)

        # Create a trajectory starting at the middle and traveling +2 pixels a day in x and -5 in y.
        trj = Trajectory(x=9, y=9, vx=2.0, vy=-5.0)

        # Predict locations at times 0.0 and 1.0
        my_sky = trajectory_predict_skypos(trj, my_wcs, [0.0, 1.0])
        self.assertAlmostEqual(my_sky.ra[0].deg, 45.0)
        self.assertAlmostEqual(my_sky.dec[0].deg, -15.0)
        self.assertAlmostEqual(my_sky.ra[1].deg, 45.2, delta=0.01)
        self.assertAlmostEqual(my_sky.dec[1].deg, -15.5, delta=0.01)

    def test_trajectory_from_np_object(self):
        np_obj = np.array(
            [(300.0, 750.0, 106.0, 44.0, 9.52, -0.5, 10.0)],
            dtype=[
                ("lh", "<f8"),
                ("flux", "<f8"),
                ("x", "<f8"),
                ("y", "<f8"),
                ("vx", "<f8"),
                ("vy", "<f8"),
                ("num_obs", "<f8"),
            ],
        )

        trj = trajectory_from_np_object(np_obj)
        self.assertEqual(trj.x, 106)
        self.assertEqual(trj.y, 44)
        self.assertAlmostEqual(trj.vx, 9.52, delta=1e-5)
        self.assertAlmostEqual(trj.vy, -0.5, delta=1e-5)
        self.assertEqual(trj.flux, 750.0)
        self.assertEqual(trj.lh, 300.0)
        self.assertEqual(trj.obs_count, 10)

    def test_trajectory_from_dict(self):
        trj_dict = {
            "x": 1,
            "y": 2,
            "vx": 3.0,
            "vy": 4.0,
            "flux": 5.0,
            "lh": 6.0,
            "obs_count": 7,
        }
        trj = trajectory_from_dict(trj_dict)

        self.assertEqual(trj.x, 1)
        self.assertEqual(trj.y, 2)
        self.assertEqual(trj.vx, 3.0)
        self.assertEqual(trj.vy, 4.0)
        self.assertEqual(trj.flux, 5.0)
        self.assertEqual(trj.lh, 6.0)
        self.assertEqual(trj.obs_count, 7)

    def test_fit_trajectory_from_pixels(self):
        x_vals = np.array([5.0, 7.0, 9.0, 11.0])
        y_vals = np.array([4.0, 3.0, 2.0, 1.0])
        times = np.array([1.0, 2.0, 3.0, 4.0])

        trj = fit_trajectory_from_pixels(x_vals, y_vals, times, centered=False)
        self.assertAlmostEqual(trj.x, 5)
        self.assertAlmostEqual(trj.y, 4)
        self.assertAlmostEqual(trj.vx, 2.0)
        self.assertAlmostEqual(trj.vy, -1.0)

        # If the pixel values are centered, we need account for the 0.5 pixel shift.
        x_vals = np.array([5.5, 7.5, 9.5, 11.5])
        y_vals = np.array([4.5, 3.5, 2.5, 1.5])
        times = np.array([1.0, 2.0, 3.0, 4.0])

        trj = fit_trajectory_from_pixels(x_vals, y_vals, times, centered=True)
        self.assertAlmostEqual(trj.x, 5)
        self.assertAlmostEqual(trj.y, 4)
        self.assertAlmostEqual(trj.vx, 2.0)
        self.assertAlmostEqual(trj.vy, -1.0)

        # We can't fit trajectories from a single point or mismatched array lengths.
        self.assertRaises(ValueError, fit_trajectory_from_pixels, [1.0], [1.0], [1.0])
        self.assertRaises(ValueError, fit_trajectory_from_pixels, [1.0, 2.0], [1.0, 2.0], [1.0])
        self.assertRaises(ValueError, fit_trajectory_from_pixels, [1.0, 2.0], [1.0], [1.0, 2.0])

    def test_trajectories_to_dict(self):
        trj_list = [
            Trajectory(x=0, y=1, vx=2.0, vy=3.0, lh=4.0, flux=5.0, obs_count=6),
            Trajectory(x=10, y=11, vx=12.0, vy=13.0, lh=14.0, flux=15.0, obs_count=16),
            Trajectory(x=20, y=21, vx=22.0, vy=23.0, lh=24.0, flux=25.0, obs_count=26),
        ]

        trj_dict = trajectories_to_dict(trj_list)
        self.assertTrue(np.array_equal(trj_dict["x"], [0, 10, 20]))
        self.assertTrue(np.array_equal(trj_dict["y"], [1, 11, 21]))
        self.assertTrue(np.array_equal(trj_dict["vx"], [2.0, 12.0, 22.0]))
        self.assertTrue(np.array_equal(trj_dict["vy"], [3.0, 13.0, 23.0]))
        self.assertTrue(np.array_equal(trj_dict["likelihood"], [4.0, 14.0, 24.0]))
        self.assertTrue(np.array_equal(trj_dict["flux"], [5.0, 15.0, 25.0]))
        self.assertTrue(np.array_equal(trj_dict["obs_count"], [6, 16, 26]))

    def test_evaluate_trajectory_mse(self):
        trj = Trajectory(x=5, y=4, vx=2.0, vy=-1.0)
        x_vals = np.array([5.5, 7.5, 9.7, 11.5])
        y_vals = np.array([4.5, 3.4, 2.5, 1.5])
        times = np.array([0.0, 1.0, 2.0, 3.0])

        mse = evaluate_trajectory_mse(trj, x_vals, y_vals, times)
        self.assertAlmostEqual(mse, (0.01 + 0.04) / 4.0)

        mse = evaluate_trajectory_mse(trj, [5.0], [4.0], [0.0], centered=False)
        self.assertAlmostEqual(mse, 0.0)

        mse = evaluate_trajectory_mse(trj, [5.5], [4.1], [0.0], centered=False)
        self.assertAlmostEqual(mse, 0.25 + 0.01)

        self.assertRaises(ValueError, evaluate_trajectory_mse, trj, [], [], [])

    def test_avg_trajectory_distance(self):
        times_0 = np.array([0.0])
        times_1 = np.array([0.0, 1.0])
        times_5 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # A trajectory is always zero pixels from itself.
        trjA = Trajectory(x=1, y=2, vx=1.0, vy=-1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjA, times_0), 0.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjA, times_1), 0.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjA, times_5), 0.0)

        # Create a trajectory with a constant 1 pixel offset in the y direction.
        trjB = Trajectory(x=1, y=1, vx=1.0, vy=-1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjB, times_0), 1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjB, times_1), 1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjB, times_5), 1.0)

        # Create a trajectory with an increasing offset in the x direction.
        trjC = Trajectory(x=1, y=2, vx=2.0, vy=-1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjC, times_0), 0.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjC, times_1), 0.5)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjC, times_5), 2.0)

        # Create a trajectory with an increasing offset in the y direction.
        trjC = Trajectory(x=1, y=2, vx=1.0, vy=1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjC, times_0), 0.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjC, times_1), 1.0)
        self.assertAlmostEqual(avg_trajectory_distance(trjA, trjC, times_5), 4.0)

        # A list of empty times is invalid.
        with self.assertRaises(ValueError):
            _ = avg_trajectory_distance(trjA, trjC, [])

    def test_match_trajectory_sets(self):
        queries = [
            Trajectory(x=0, y=0, vx=0.0, vy=0.0),
            Trajectory(x=10, y=10, vx=0.5, vy=-2.0),
            Trajectory(x=50, y=80, vx=-1.0, vy=0.0),
        ]
        candidates = [
            Trajectory(x=0, y=0, vx=0.0, vy=0.0),  # Same as queries[0]
            Trajectory(x=49, y=82, vx=-1.0, vy=0.01),  # Close to queries[2]
        ]
        results = match_trajectory_sets(queries, candidates, 5.0, [0.0, 10.0])
        self.assertTrue(np.array_equal(results, [0, -1, 1]))

        # Add a trajectory that is too far from queries[1] to be a good match.
        candidates.append(Trajectory(x=15, y=15, vx=0.5, vy=-2.0))
        results = match_trajectory_sets(queries, candidates, 5.0, [0.0, 10.0])
        self.assertTrue(np.array_equal(results, [0, -1, 1]))

        # Add a trajectory that is close to queries[1].
        candidates.append(Trajectory(x=10, y=10, vx=0.6, vy=-2.5))
        results = match_trajectory_sets(queries, candidates, 5.0, [0.0, 10.0])
        self.assertTrue(np.array_equal(results, [0, 3, 1]))

        # Add a trajectory that is even closer to queries[1].
        candidates.append(Trajectory(x=10, y=10, vx=0.6, vy=-2.1))
        results = match_trajectory_sets(queries, candidates, 5.0, [0.0, 10.0])
        self.assertTrue(np.array_equal(results, [0, 4, 1]))

        # Add another query trajectory that is close to queries[0], but
        # not close enough to steal its match.
        queries.append(Trajectory(x=1, y=0, vx=0.0, vy=0.0))
        results = match_trajectory_sets(queries, candidates, 5.0, [0.0, 10.0])
        self.assertTrue(np.array_equal(results, [0, 4, 1, -1]))

        # Add another trajectory that is close to queries[0], but not as close
        # as its current match. So this gets matched with queries[3] instead.
        candidates.append(Trajectory(x=0, y=0, vx=0.0, vy=0.01))
        results = match_trajectory_sets(queries, candidates, 5.0, [0.0, 10.0])
        self.assertTrue(np.array_equal(results, [0, 4, 1, 5]))


if __name__ == "__main__":
    unittest.main()
