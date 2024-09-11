"""
This is a manually run regression test that is more comprehensive than
the individual unittests.
"""

import logging
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
import astropy.wcs

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data.fake_data_creator import add_fake_object, make_fake_layered_image
from kbmod.file_utils import *
from kbmod.results import Results
from kbmod.run_search import SearchRunner
from kbmod.search import *
from kbmod.wcs_utils import make_fake_wcs_info
from kbmod.work_unit import WorkUnit

logger = logging.getLogger(__name__)


def ave_trajectory_distance(trjA, trjB, times=[0.0]):
    """Evaluate the average distance between two trajectories (in pixels)
    at different times.

    Parameters
    ----------
    trjA : `kbmod.search.Trajectory`
        The first Trajectory to evaluate.
    trjB : `kbmod.search.Trajectory`
        The second Trajectory to evaluate.
    times : `list`
        The list of zero-shifted times at which to evaluate the
        matches. The average of the distances at these times
        are used.

    Returns
    -------
    ave_dist : `float`
        The average distance in pixels.
    """
    total = 0.0
    for t in times:
        dx = (trjA.x + t * trjA.vx) - (trjB.x + t * trjB.vx)
        dy = (trjA.y + t * trjA.vy) - (trjB.y + t * trjB.vy)
        total += math.sqrt(dx * dx + dy * dy)

    ave_dist = total / len(times)
    return ave_dist


def find_unique_overlap(traj_query, traj_base, threshold, times=[0.0]):
    """Finds the set of trajectories in traj_query that are 'close' to
    trajectories in traj_base such that each Trajectory in traj_base
    is used at most once.

    Used to evaluate the performance of algorithms.

    Parameters
    ----------
    traj1 : `list`
        A list of trajectories to compare.
    traj2 : `list`
        The second list of trajectories to compare.
    threshold : float
        The distance threshold between two observations to count a
        match (in pixels).
    times : `list`
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    results : `list`
        The list of trajectories that appear in both traj1 and traj2
        where each Trajectory in each set is only used once.
    """
    num_times = len(times)
    size_base = len(traj_base)
    used = [False] * size_base

    results = []
    for query in traj_query:
        best_dist = 10.0 * threshold
        best_ind = -1

        # Check the current query against all unused base trajectories.
        for j in range(size_base):
            if not used[j]:
                dist = ave_trajectory_distance(query, traj_base[j], times)
                if dist < best_dist:
                    best_dist = dist
                    best_ind = j

        # If we found a good match, save it.
        if best_dist <= threshold:
            results.append(query)
            used[best_ind] = True
    return results


def find_set_difference(traj_query, traj_base, threshold, times=[0.0]):
    """Finds the set of trajectories in traj_query that are NOT 'close' to
    any trajectories in traj_base such that each Trajectory in traj_base
    is used at most once.

    Used to evaluate the performance of algorithms.

    Parameters
    ----------
    traj_query : `list`
        A list of trajectories to compare.
    traj_base : `list`
        The second list of trajectories to compare.
    threshold : `float`
        The distance threshold between two observations
        to count a match (in pixels).
    times : `list`
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    results : `list`
        A list of trajectories that appear in traj_query but not
        in traj_base where each Trajectory in each set is only
        used once.
    """
    num_times = len(times)
    size_base = len(traj_base)
    used = [False] * size_base

    results = []
    for query in traj_query:
        best_dist = 10.0 * threshold
        best_ind = -1

        # Check the current query against all unused base trajectories.
        for j in range(size_base):
            if not used[j]:
                dist = ave_trajectory_distance(query, traj_base[j], times)
                if dist < best_dist:
                    best_dist = dist
                    best_ind = j

        # If we found a good match, save it.
        if best_dist <= threshold:
            used[best_ind] = True
        else:
            results.append(query)
    return results


def make_fake_ImageStack(times, trjs, psf_vals):
    """Make a stack of fake layered images.

    Parameters
    ----------
    times : `list`
        A list of time stamps.
    trjs : `list`
        A list of trajectories.
    psf_vals : `list`
        A list of PSF variances.

    Returns
    -------
        A ImageStack
    """
    imCount = len(times)
    t0 = times[0]
    dim_x = 512
    dim_y = 1024
    noise_level = 4.0
    variance = noise_level**2

    imlist = []
    for i in range(imCount):
        p = PSF(psf_vals[i])
        time = times[i] - t0

        img = make_fake_layered_image(dim_x, dim_y, noise_level, variance, times[i], p, seed=i)

        for trj in trjs:
            px = trj.x + time * trj.vx + 0.5
            py = trj.y + time * trj.vy + 0.5
            add_fake_object(img, px, py, trj.flux, p)

        imlist.append(img)
    stack = ImageStack(imlist)
    return stack


def load_trajectories_from_file(filename):
    """Load in the result trajectories from their file.

    Parameters
    ----------
    filename : `str`
         The path and filename of the results.

    Returns
    -------
    trjs : `list`
        The list of trajectories
    """
    trjs = []
    res_new = np.loadtxt(filename, dtype=str)
    for i in range(len(res_new)):
        x = int(res_new[i][5])
        y = int(res_new[i][7])
        xv = float(res_new[i][9])
        yv = float(res_new[i][11])
        flux = float(res_new[i][3])
        trjs.append(Trajectory(x, y, xv, yv, flux))
    return trjs


def perform_search(im_stack, res_filename, default_psf):
    """
    Run the core search algorithm.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images to search.
    res_filename : `str`
        The path (directory) for the new result files.
    default_psf : `float`
        The default PSF value to use when nothing is provided
        in the PSF file.
    """
    v_min = 92.0  # Pixels/day
    v_max = 550.0

    # Manually set the average angle that will work with the (manually specified) tracks.
    average_angle = 1.1901106654050821

    # Offset by PI for prograde orbits in lori allen data
    ang_below = -np.pi + np.pi / 10.0  # Angle below ecliptic
    ang_above = np.pi + np.pi / 10.0  # Angle above ecliptic
    v_steps = 51
    ang_steps = 25

    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]
    num_obs = 15

    input_parameters = {
        "im_filepath": "./",
        "res_filepath": None,
        "result_filename": res_filename,
        "psf_val": default_psf,
        "output_suffix": "",
        "v_arr": v_arr,
        "average_angle": average_angle,
        "ang_arr": ang_arr,
        "num_obs": num_obs,
        "do_mask": True,
        "lh_level": 25.0,
        "sigmaG_lims": [25, 75],
        "mom_lims": [37.5, 37.5, 1.5, 1.0, 1.0],
        "peak_offset": [3.0, 3.0],
        "chunk_size": 1000000,
        "stamp_type": "cpp_median",
        "eps": 20.0,
        "gpu_filter": True,
        "clip_negative": True,
        "x_pixel_buffer": 10,
        "y_pixel_buffer": 10,
        "debug": True,
    }
    config = SearchConfiguration.from_dict(input_parameters)

    wu = WorkUnit(im_stack=im_stack, config=config)  # , wcs=fake_wcs)
    rs = SearchRunner()
    rs.run_search_from_work_unit(wu)


def run_full_test():
    """Run the full test.

    Returns
    -------
    A bool indicating whether the test was successful.
    """
    default_psf = 1.05

    # Used a fixed set of trajectories so we always know the ground truth.
    flux_val = 500.0
    trjs = [
        Trajectory(357, 997, -15.814404, -172.098450, flux_val),
        Trajectory(477, 777, -70.858154, -117.137817, flux_val),
        Trajectory(408, 533, -53.721024, -106.118118, flux_val),
        Trajectory(425, 740, -32.865086, -132.898575, flux_val),
        Trajectory(489, 881, -73.831688, -93.251732, flux_val),
        Trajectory(412, 980, -79.985207, -192.813080, flux_val),
        Trajectory(443, 923, -36.977375, -103.556976, flux_val),
        Trajectory(368, 1015, -43.644382, -176.487488, flux_val),
        Trajectory(510, 1011, -125.422997, -166.863983, flux_val),
        Trajectory(398, 939, -51.037308, -107.434616, flux_val),
        Trajectory(491, 925, -74.266739, -104.155556, flux_val),
        Trajectory(366, 824, -18.041782, -153.808197, flux_val),
        Trajectory(477, 870, -45.608849, -90.093689, flux_val),
        Trajectory(447, 993, -38.152031, -196.087646, flux_val),
        Trajectory(481, 882, -96.767357, -143.192352, flux_val),
        Trajectory(423, 912, -104.900154, -125.859169, flux_val),
        Trajectory(409, 803, -99.066856, -173.469589, flux_val),
        Trajectory(328, 797, -33.212299, -196.984467, flux_val),
        Trajectory(466, 1026, -67.892105, -118.881493, flux_val),  # Off chip y
        Trajectory(514, 795, -20.134245, -171.646683, flux_val),  # Off chip x
    ]

    with tempfile.TemporaryDirectory() as dir_name:
        # Generate the fake data - 'obs_per_night' observations a night,
        # spaced ~15 minutes apart.
        num_times = 20
        times = []
        psf_vals = []
        seen_on_day = 0
        day_num = 0
        for i in range(num_times):
            t = 57130.2 + day_num + seen_on_day * 0.01
            times.append(t)

            seen_on_day += 1
            if seen_on_day == 4:
                seen_on_day = 0
                day_num += 1

            # Set PSF values between +/- 0.1 around the default value.
            psf_vals.append(default_psf - 0.1 + 0.1 * (i % 3))

        stack = make_fake_ImageStack(times, trjs, psf_vals)

        # Do the search.
        result_filename = os.path.join(dir_name, "results.ecsv")
        perform_search(stack, result_filename, default_psf)

        # Load the results from the results file and extract a list of trajectories.
        loaded_data = Results.read_table(result_filename)
        found = loaded_data.make_trajectory_list()
        logger.debug("Found %i trajectories vs %i used." % (len(found), len(trjs)))

        # Determine which trajectories we did not recover.
        overlap = find_unique_overlap(trjs, found, 3.0, [0.0, 2.0])
        missing = find_set_difference(trjs, found, 3.0, [0.0, 2.0])

        logger.debug("\nRecovered %i matching trajectories:" % len(overlap))
        for x in overlap:
            logger.debug(x)

        if len(missing) == 0:
            logger.debug("*** PASSED ***")
            return True
        else:
            logger.debug("\nFailed to recover %i trajectories:" % len(missing))
            for x in missing:
                logger.debug(x)
            logger.debug("*** FAILED ***")
            return False


# The unit test runner
class test_regression_test(unittest.TestCase):
    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_run_test(self):
        self.assertTrue(run_full_test())


if __name__ == "__main__":
    unittest.main()
