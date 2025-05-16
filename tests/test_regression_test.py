"""
This is a manually run regression test that is more comprehensive than
the individual unittests.
"""

import logging
import os
import tempfile
import unittest
import warnings

import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import image_stack_add_fake_object, make_fake_image_stack
from kbmod.core.psf import PSF
from kbmod.image_utils import image_stack_py_to_cpp
from kbmod.results import Results
from kbmod.run_search import SearchRunner
from kbmod.search import *
from kbmod.trajectory_utils import match_trajectory_sets
from kbmod.work_unit import WorkUnit

logger = logging.getLogger(__name__)


def make_fake_images(times, trjs, psf_vals):
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

    # Create the array of PSF kernels.
    psfs = [PSF.make_gaussian_kernel(psf_vals[i]) for i in range(imCount)]

    # Create the data, including fake objects.
    rng = np.random.default_rng(1001)
    stack = make_fake_image_stack(dim_y, dim_x, times, noise_level=4.0, psfs=psfs, rng=rng)
    for trj in trjs:
        image_stack_add_fake_object(stack, trj.x, trj.y, trj.vx, trj.vy, trj.flux)
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
    input_parameters = {
        "result_filename": res_filename,
        "psf_val": default_psf,
        "generator_config": {
            "name": "EclipticCenteredSearch",
            # Offset by PI for prograde orbits in lori allen data
            "angles": [np.pi - np.pi / 10.0, np.pi + np.pi / 10.0, 26],
            "velocities": [92.0, 550.0, 52],
            "angle_units": "radian",
            "given_ecliptic": 1.1901106654050821,
        },
        "num_obs": 15,
        "do_mask": True,
        "lh_level": 25.0,
        "sigmaG_lims": [25, 75],
        "chunk_size": 1000000,
        "stamp_type": "cpp_median",
        "cluster_eps": 20.0,
        "gpu_filter": True,
        "clip_negative": True,
        "x_pixel_buffer": 10,
        "y_pixel_buffer": 10,
        "debug": False,
    }
    config = SearchConfiguration.from_dict(input_parameters)

    # Create fake visit metadata to confirm we pass it along.  We catch the
    # warning about missing WCS.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wu = WorkUnit(im_stack=im_stack, config=config)
    wu.org_img_meta["visit"] = [f"img_{i}" for i in range(im_stack.num_times)]

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
        Trajectory(515, 881, -73.831688, -93.251732, flux_val),
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

        stack_py = make_fake_images(times, trjs, psf_vals)
        stack = image_stack_py_to_cpp(stack_py)

        # Do the search.
        result_filename = os.path.join(dir_name, "results.ecsv")
        perform_search(stack, result_filename, default_psf)

        # Load the results from the results file and extract a list of trajectories.
        loaded_data = Results.read_table(result_filename)
        found = loaded_data.make_trajectory_list()
        logger.debug("Found %i trajectories vs %i used." % (len(found), len(trjs)))

        logger.debug("Used trajectories:")
        for x in trjs:
            logger.debug(x)
        logger.debug("Found trajectories:")
        for x in found:
            logger.debug(x)

        # Check that we saved the correct meta data for the table.
        assert loaded_data.table.meta["num_img"] == num_times
        assert loaded_data.table.meta["dims"] == (stack.width, stack.height)
        assert np.allclose(loaded_data.table.meta["mjd_mid"], times)
        assert np.array_equal(
            loaded_data.table.meta["visit"],
            [f"img_{i}" for i in range(stack.num_times)],
        )

        # Determine which trajectories we did not recover.
        matches = match_trajectory_sets(trjs, found, 3.0, [0.0, 2.0])
        overlap = np.where(matches > -1)[0]
        missing = np.where(matches == -1)[0]

        logger.debug("\nRecovered %i matching trajectories:" % len(overlap))
        for x in overlap:
            logger.debug(trjs[x])

        if len(missing) == 0:
            logger.debug("*** PASSED ***")
            return True
        else:
            logger.debug("\nFailed to recover %i trajectories:" % len(missing))
            for x in missing:
                logger.debug(trjs[x])
            logger.debug("*** FAILED ***")
            return False


# The unit test runner
class test_regression_test(unittest.TestCase):
    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_run_test(self):
        self.assertTrue(run_full_test())


if __name__ == "__main__":
    unittest.main()
