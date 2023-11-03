"""
This is a manually run regression test that is more comprehensive than
the individual unittests.
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from kbmod.fake_data_creator import add_fake_object
from kbmod.file_utils import *
from kbmod.run_search import SearchRunner
from kbmod.search import *


def ave_trajectory_distance(trjA, trjB, times=[0.0]):
    """Evaluate the average distance between two trajectories (in pixels)
    at different times.

    Parameters
    ----------
    trjA : `trajectory`
        The first Trajectory to evaluate.
    trjB : `trajectory`
        The second Trajectory to evaluate.
    times : list
        The list of zero-shifted times at which to evaluate the
        matches. The average of the distances at these times
        are used.

    Returns
    -------
    ave_dist : float
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
    traj1 : list
        A list of trajectories to compare.
    traj2 : list
        The second list of trajectories to compare.
    threshold : float
        The distance threshold between two observations to count a
        match (in pixels).
    times : list
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    results : list
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
    traj_query : list
        A list of trajectories to compare.
    traj_base : list
        The second list of trajectories to compare.
    threshold : float
        The distance threshold between two observations
        to count a match (in pixels).
    times : list
        The list of zero-shifted times at which to evaluate the matches.
        The average of the distances at these times are used.

    Returns
    -------
    results : list
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


def make_trajectory(x, y, vx, vy, flux):
    """Create a fake Trajectory given the parameters.

    Arguments:
        x : int
            The starting x coordinate.
        y : int
            The starting y coordinate.
        vx : float
            The velocity in x.
        vy : float
            The velocity in y.
        flux : float
            The flux of the object.

    Returns:
        A Trajectory object.
    """
    t = Trajectory()
    t.x = x
    t.y = y
    t.vx = vx
    t.vy = vy
    t.flux = flux
    return t


def make_fake_ImageStack(times, trjs, psf_vals):
    """
    Make a stack of fake layered images.

    Arguments:
        times : list
            A list of time stamps.
        trjs : list
            A list of trajectories.
        psf_vals : list
            A list of PSF variances.

    Returns:
        A ImageStack
    """
    imCount = len(times)
    t0 = times[0]
    dim_x = 512
    dim_y = 1024
    noise_level = 8.0
    variance = noise_level**2

    imlist = []
    for i in range(imCount):
        p = PSF(psf_vals[i])
        time = times[i] - t0

        # For each odd case, don't save the time. These will be provided by the time file.
        saved_time = times[i]
        if i % 2 == 1:
            saved_time = 0.0

        img = LayeredImage(("%06i" % i), dim_x, dim_y, noise_level, variance, saved_time, p, i)

        for trj in trjs:
            px = trj.x + time * trj.vx + 0.5
            py = trj.y + time * trj.vy + 0.5
            add_fake_object(img, px, py, trj.flux, p)

        imlist.append(img)
    stack = ImageStack(imlist)
    return stack


def add_wcs_header_data(full_file_name):
    """
    Add (fixed) WCS data to a fits file.

    Arguments:
        full_file_name : string
    """
    hdul = fits.open(full_file_name)
    hdul[1].header["WCSAXES"] = 2
    hdul[1].header["CTYPE1"] = "RA---TAN-SIP"
    hdul[1].header["CTYPE2"] = "DEC--TAN-SIP"
    hdul[1].header["CRVAL1"] = 200.614997245422
    hdul[1].header["CRVAL2"] = -7.78878863332778
    hdul[1].header["CRPIX1"] = 1033.934327
    hdul[1].header["CRPIX2"] = 2043.548284
    hdul[1].header["CD1_1"] = -1.13926485986789e-07
    hdul[1].header["CD1_2"] = 7.31839748843125e-05
    hdul[1].header["CD2_1"] = -7.30064978350695e-05
    hdul[1].header["CD2_2"] = -1.27520156332774e-07
    hdul[1].header["CTYPE1A"] = "LINEAR  "
    hdul[1].header["CTYPE2A"] = "LINEAR  "
    hdul[1].header["CUNIT1A"] = "PIXEL   "
    hdul[1].header["CUNIT2A"] = "PIXEL   "
    hdul.writeto(full_file_name, overwrite=True)


def save_fake_data(data_dir, stack, times, psf_vals, default_psf_val=1.0):
    # Make the subdirectory if needed.
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        print("Directory '%s' does not exist. Creating." % data_dir)
        os.mkdir(data_dir)

    # Make the subdirectory if needed.
    img_dir = data_dir + "/imgs"
    dir_path = Path(img_dir)
    if not dir_path.is_dir():
        print("Directory '%s' does not exist. Creating." % img_dir)
        os.mkdir(img_dir)

    # Save each of the image files.
    for i in range(stack.img_count()):
        img = stack.get_single_image(i)
        filename = img_dir + "/" + img.get_name() + ".fits"
        print("Saving file: %s" % filename)

        # If the file already exists, delete it.
        if Path(filename).exists():
            os.remove(filename)

        # Save the file.
        img.save_layers(img_dir + "/")

        # Open the file and insert fake WCS data.
        add_wcs_header_data(filename)

    # Save the psf file.
    psf_file_name = data_dir + "/psf_vals.dat"
    print("Creating psf file: %s" % psf_file_name)
    with open(psf_file_name, "w") as file:
        file.write("# visit_id psf_val\n")
        for i in range(len(times)):
            if psf_vals[i] != default_psf_val:
                file.write("%06i %f\n" % (i, psf_vals[i]))

    # Save the time file, but only include half the file times (odd indices).
    time_file_name = data_dir + "/times.dat"
    time_mapping = {}
    for i in range(len(times)):
        if i % 2 == 1:
            id_str = stack.get_single_image(i).get_name()
            time_mapping[id_str] = times[i]
    FileUtils.save_time_dictionary(time_file_name, time_mapping)


def load_trajectories_from_file(filename):
    """
    Load in the result trajectories from their file.

    Arguments:
         filename - The path and filename of the results.

    Returns:
         list : a list of trajectories
    """
    trjs = []
    res_new = np.loadtxt(filename, dtype=str)
    for i in range(len(res_new)):
        x = int(res_new[i][5])
        y = int(res_new[i][7])
        xv = float(res_new[i][9])
        yv = float(res_new[i][11])
        flux = float(res_new[i][3])
        trjs.append(make_trajectory(x, y, xv, yv, flux))
    return trjs


def perform_search(im_filepath, time_file, psf_file, res_filepath, results_suffix, default_psf):
    """
    Run the core search algorithm.

    Arguments:
      im_filepath - The file path (directory) for the image files.
      time_file - The path and file name of the file of timestamps.
      psf_file - The path and file name of the psf values.
      res_filepath - The path (directory) for the new result files.
      results_suffix - The file suffix to use for the new results.
      default_psf - The default PSF value to use when nothing is provided
                    in the PSF file.
    """
    v_min = 92.0  # Pixels/day
    v_max = 550.0

    # Offset by PI for prograde orbits in lori allen data
    ang_below = -np.pi + np.pi / 10.0  # Angle below ecliptic
    ang_above = np.pi + np.pi / 10.0  # Angle above ecliptic
    v_steps = 51
    ang_steps = 25

    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]
    num_obs = 15

    mask_bits_dict = {
        "BAD": 0,
        "CLIPPED": 9,
        "CR": 3,
        "DETECTED": 5,
        "DETECTED_NEGATIVE": 6,
        "EDGE": 4,
        "INEXACT_PSF": 10,
        "INTRP": 2,
        "NOT_DEBLENDED": 11,
        "NO_DATA": 8,
        "REJECTED": 12,
        "SAT": 1,
        "SENSOR_EDGE": 13,
        "SUSPECT": 7,
    }
    flag_keys = [
        "BAD",
        "CR",
        "INTRP",
        "NO_DATA",
        "SENSOR_EDGE",
        "SAT",
        "SUSPECT",
        "CLIPPED",
        "REJECTED",
        "DETECTED_NEGATIVE",
    ]
    repeated_flag_keys = ["DETECTED"]

    input_parameters = {
        "im_filepath": im_filepath,
        "res_filepath": res_filepath,
        "time_file": time_file,
        "psf_file": psf_file,
        "psf_val": default_psf,
        "output_suffix": results_suffix,
        "v_arr": v_arr,
        "ang_arr": ang_arr,
        "num_obs": num_obs,
        "do_mask": True,
        "lh_level": 25.0,
        "mjd_lims": [52130.0, 62130.0],
        "sigmaG_lims": [25, 75],
        "mom_lims": [37.5, 37.5, 1.5, 1.0, 1.0],
        "peak_offset": [3.0, 3.0],
        "chunk_size": 1000000,
        "stamp_type": "cpp_median",
        "eps": 0.03,
        "gpu_filter": True,
        "clip_negative": True,
        "mask_num_images": 10,
        "mask_bits_dict": mask_bits_dict,
        "flag_keys": flag_keys,
        "repeated_flag_keys": repeated_flag_keys,
        "x_pixel_buffer": 10,
        "y_pixel_buffer": 10,
        "debug": True,
    }

    rs = SearchRunner()
    rs.run_search_from_config(input_parameters)


if __name__ == "__main__":
    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_times", default=20, help="The number of time steps to use.")
    parser.add_argument("--obs_per_night", default=4, help="The number of same night observations.")
    parser.add_argument("--flux", default=500.0, help="The flux level to use.")
    parser.add_argument("--default_psf", default=1.05, help="The default PSF value to use.")
    args = parser.parse_args()
    default_psf = float(args.default_psf)

    # Used a fixed set of trajectories so we always know the ground truth.
    flux_val = float(args.flux)
    trjs = [
        make_trajectory(357, 997, -15.814404, -172.098450, flux_val),
        make_trajectory(477, 777, -70.858154, -117.137817, flux_val),
        make_trajectory(408, 533, -53.721024, -106.118118, flux_val),
        make_trajectory(425, 740, -32.865086, -132.898575, flux_val),
        make_trajectory(489, 881, -73.831688, -93.251732, flux_val),
        make_trajectory(412, 980, -79.985207, -192.813080, flux_val),
        make_trajectory(443, 923, -36.977375, -103.556976, flux_val),
        make_trajectory(368, 1015, -43.644382, -176.487488, flux_val),
        make_trajectory(510, 1011, -125.422997, -166.863983, flux_val),
        make_trajectory(398, 939, -51.037308, -107.434616, flux_val),
        make_trajectory(491, 925, -74.266739, -104.155556, flux_val),
        make_trajectory(366, 824, -18.041782, -153.808197, flux_val),
        make_trajectory(477, 870, -45.608849, -90.093689, flux_val),
        make_trajectory(447, 993, -38.152031, -196.087646, flux_val),
        make_trajectory(481, 882, -96.767357, -143.192352, flux_val),
        make_trajectory(423, 912, -104.900154, -125.859169, flux_val),
        make_trajectory(409, 803, -99.066856, -173.469589, flux_val),
        make_trajectory(328, 797, -33.212299, -196.984467, flux_val),
        make_trajectory(466, 1026, -67.892105, -118.881493, flux_val),  # Off chip y
        make_trajectory(514, 795, -20.134245, -171.646683, flux_val),  # Off chip x
    ]

    with tempfile.TemporaryDirectory() as dir_name:
        # Generate the fake data - 'obs_per_night' observations a night,
        # spaced ~15 minutes apart.
        num_times = int(args.num_times)
        times = []
        psf_vals = []
        seen_on_day = 0
        day_num = 0
        for i in range(num_times):
            t = 57130.2 + day_num + seen_on_day * 0.01
            times.append(t)

            seen_on_day += 1
            if seen_on_day == args.obs_per_night:
                seen_on_day = 0
                day_num += 1

            # Set PSF values between +/- 0.1 around the default value.
            psf_vals.append(default_psf - 0.1 + 0.1 * (i % 3))

        # Add several instances to the end that will be filtered by the time bounds.
        for i in range(3):
            times.append(67130.2 + i)
            psf_vals.append(default_psf + 0.01)

        stack = make_fake_ImageStack(times, trjs, psf_vals)
        save_fake_data(dir_name, stack, times, psf_vals, default_psf)

        # Do the search.
        print("Running search with data in %s/" % dir_name)
        perform_search(
            dir_name + "/imgs",
            dir_name + "/times.dat",
            dir_name + "/psf_vals.dat",
            dir_name,
            "tmp",
            default_psf,
        )

        # Load the results from the results file.
        found = load_trajectories_from_file(dir_name + "/results_tmp.txt")
        print("Found %i trajectories vs %i used." % (len(found), len(trjs)))

        # Determine which trajectories we did not recover.
        overlap = find_unique_overlap(trjs, found, 3.0, [0.0, 2.0])
        missing = find_set_difference(trjs, found, 3.0, [0.0, 2.0])

        print("\nRecovered %i matching trajectories:" % len(overlap))
        for x in overlap:
            print(x)

        if len(missing) == 0:
            print("*** PASSED ***")
        else:
            print("\nFailed to recover %i trajectories:" % len(missing))
            for x in missing:
                print(x)
            print("*** FAILED ***")
