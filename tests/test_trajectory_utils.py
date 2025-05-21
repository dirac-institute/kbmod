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

        # Predict locations at times 57921.0 and 57922.0, both as a list and as a numpy array.
        obstimes = [57921.0, 57922.0]
        for curr_obstimes in [obstimes, np.array(obstimes)]:
            my_sky = trajectory_predict_skypos(trj, my_wcs, curr_obstimes)
            # Verify that the obstimes were not mutated
            self.assertEqual(curr_obstimes[0], 57921.0)
            self.assertEqual(curr_obstimes[1], 57922.0)

            # Verify that the predicted sky positions are correct.
            self.assertAlmostEqual(my_sky.ra[0].deg, 45.0)
            self.assertAlmostEqual(my_sky.dec[0].deg, -15.0)
            self.assertAlmostEqual(my_sky.ra[1].deg, 45.2, delta=0.01)
            self.assertAlmostEqual(my_sky.dec[1].deg, -15.5, delta=0.01)

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

    def test_find_closest_trajectory(self):
        candidates = [
            Trajectory(x=0, y=0, vx=0.0, vy=0.0),
            Trajectory(x=10, y=10, vx=0.5, vy=-2.0),
            Trajectory(x=49, y=82, vx=-1.0, vy=0.01),
            Trajectory(x=50, y=80, vx=-1.0, vy=0.0),
            Trajectory(x=100, y=100, vx=1.0, vy=1.0),
        ]

        # Exact match with candidate 0.
        idx, dist = find_closest_trajectory(
            Trajectory(x=0, y=0, vx=0.0, vy=0.0),
            candidates,
            [0.0, 10.0],
        )
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(dist, 0.0)

        # Close match with candidate 2.
        idx, dist = find_closest_trajectory(
            Trajectory(x=48, y=83, vx=-1.01, vy=0.0),
            candidates,
            [0.0, 10.0],
        )
        self.assertEqual(idx, 2)
        self.assertAlmostEqual(dist, 1.4177402651666815)

        # Not really close match to 1.
        idx, dist = find_closest_trajectory(
            Trajectory(x=20, y=20, vx=-0.5, vy=1.0),
            candidates,
            [0.0, 10.0],
        )
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(dist, 27.071067811865476)

        # We match something even when nothing is close
        idx, dist = find_closest_trajectory(
            Trajectory(x=2000, y=2000, vx=10.5, vy=15.0),
            candidates,
            [0.0, 10.0],
        )
        self.assertEqual(idx, 4)
        self.assertGreater(dist, 100.0)

    def test_find_closest_velocity(self):
        candidates = [
            Trajectory(x=0, y=0, vx=0.0, vy=0.0),
            Trajectory(x=10, y=10, vx=0.5, vy=-2.0),
            Trajectory(x=49, y=82, vx=-1.0, vy=0.01),
            Trajectory(x=50, y=80, vx=-1.0, vy=0.0),
            Trajectory(x=100, y=100, vx=1.0, vy=1.0),
        ]

        # Exact match with candidate 0.
        idx = find_closest_velocity(Trajectory(x=500, y=500, vx=0.0, vy=0.0), candidates)
        self.assertEqual(idx, 0)

        # Close match with candidate 1.
        idx = find_closest_velocity(Trajectory(x=1000, y=1000, vx=0.49, vy=-1.99), candidates)
        self.assertEqual(idx, 1)

        # Close match with candidate 3.
        idx = find_closest_velocity(Trajectory(x=1000, y=1000, vx=-1.0, vy=-0.01), candidates)
        self.assertEqual(idx, 3)

        # Far match with candidate 4.
        idx = find_closest_velocity(Trajectory(x=1000, y=1000, vx=10.0, vy=10.0), candidates)
        self.assertEqual(idx, 4)


if __name__ == "__main__":
    unittest.main()
