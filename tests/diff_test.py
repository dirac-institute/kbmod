import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from kbmod.run_search import run_search


def check_and_create_goldens_dir():
    """
    Test whether the goldens directory exists and create it if not.
    """
    dir_path = Path("goldens")
    if not dir_path.is_dir():
        os.mkdir("goldens")


def check_goldens_exist(results_suffix):
    """
    Test whether the needed goldens files exist.

    Arguments:
        results_suffix - The suffix of the goldens file.
    """
    file_path = Path("goldens/results_%s.txt" % results_suffix)
    if not file_path.is_file():
        return False

    file_path = Path("goldens/ps_%s.txt" % results_suffix)
    if not file_path.is_file():
        return False
    return True


def compare_ps_files(goldens_file, new_results_file, delta=0.00001):
    """
    Compare two PS result files.

    Arguments:
         goldens_file - The path and filename of the golden results.
         new_results_file - The path and filename of the new results.
         delta - The maximum difference in any entry.

    Returns:
         A Boolean indicating whether the files are the same.
    """
    files_equal = True

    res_new = np.loadtxt(new_results_file, dtype=str)
    print("Loaded %i new results from %s." % (len(res_new), new_results_file))
    res_old = np.loadtxt(goldens_file, dtype=str)
    print("Loaded %i old results from %s." % (len(res_old), goldens_file))

    # Check that the number of results matches up.
    if len(res_new) != len(res_old):
        print("Mismatched number of results (%i vs %i)." % (len(res_old), len(res_new)))
        files_equal = False
    else:
        # Check each line to see if it matches.
        for i in range(len(res_new)):
            old_line = res_old[i]
            new_line = res_new[i]

            found_diff = False
            if len(new_line) != len(old_line):
                found_diff = True
            else:
                for d in range(len(new_line)):
                    if (abs(float(old_line[d]) - float(new_line[d]))) > delta:
                        found_diff = True

            if found_diff:
                files_equal = False
                print("Found a difference in line %i:" % i)
                print("  [OLD] %s" % old_line[i])
                print("  [NEW] %s" % new_line[i])
    return files_equal


def compare_result_files(goldens_file, new_results_file, delta=0.001):
    """
    Compare two result files.

    Arguments:
         goldens_file - The path and filename of the golden results.
         new_results_file - The path and filename of the new results.
         delta - The maximum difference in numerical values.

    Returns:
         A Boolean indicating whether the files are the same.
    """
    files_equal = True

    res_new = np.loadtxt(new_results_file, dtype=str)
    print("Loaded %i new results from %s." % (len(res_new), new_results_file))
    res_old = np.loadtxt(goldens_file, dtype=str)
    print("Loaded %i old results from %s." % (len(res_old), goldens_file))

    # Check that the number of results matches up.
    if len(res_new) != len(res_old):
        print("Mismatched number of results (%i vs %i)." % (len(res_old), len(res_new)))
        files_equal = False
    else:
        # Check each line to see if it matches.
        for i in range(len(res_new)):
            old_line = res_old[i]
            new_line = res_new[i]

            found_diff = False
            if len(new_line) != len(old_line):
                found_diff = True
            else:
                for d in range(len(new_line)):
                    if d % 2 == 1:
                        # Allow a small difference in estimated velocities because
                        # the search might find multiple results whose scores tie.
                        if abs(float(old_line[d]) - float(new_line[d])) > delta:
                            found_diff = True
                    else:
                        if old_line[d] != new_line[d]:
                            found_diff = True

            if found_diff:
                files_equal = False
                print("Found a difference in line %i:" % i)
                print("  [OLD] %s" % old_line)
                print("  [NEW] %s" % new_line)
    return files_equal


def perform_search(im_filepath, time_file, psf_file, res_filepath, res_suffix, shorten_search=False):
    """
    Run the core search algorithm.

    Arguments:
      im_filepath - The file path (directory) for the image files.
      time_file - The path and file name of the file of timestamps.
      res_filepath - The path (directory) for the new result files.
      res_suffix - The file suffix to use for the new results.
      shorten_search - A Boolean indicating whether to use
          a coarser grid for a fast but less thorough search.
    """
    v_min = 92.0  # Pixels/day
    v_max = 550.0

    # Offset by PI for prograde orbits in lori allen data
    ang_below = -np.pi + np.pi / 10.0  # Angle below ecliptic
    ang_above = np.pi + np.pi / 10.0  # Angle above ecliptic
    v_steps = 512
    ang_steps = 256
    if shorten_search:
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
        "output_suffix": results_suffix,
        "v_arr": v_arr,
        "ang_arr": ang_arr,
        "num_obs": num_obs,
        "do_mask": True,
        "lh_level": 10.0,
        "sigmaG_lims": [25, 75],
        "mom_lims": [37.5, 37.5, 1.5, 1.0, 1.0],
        "peak_offset": [3.0, 3.0],
        "chunk_size": 1000000,
        "stamp_type": "parallel_sum",
        "eps": 0.03,
        "gpu_filter": True,
        "clip_negative": True,
        "mask_num_images": 10,
        "sigmaG_filter_type": "both",
        "mask_bits_dict": mask_bits_dict,
        "flag_keys": flag_keys,
        "repeated_flag_keys": repeated_flag_keys,
        "known_obj_thresh": None,
        "known_obj_jpl": False,
        "encode_psi_bytes": -1,
        "encode_phi_bytes": -1,
    }

    rs = run_search(input_parameters)
    rs.run_search()


if __name__ == "__main__":
    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_goldens",
        default=False,
        action="store_true",
        help="Generate the golden files (WARNING overwrites files).",
    )
    parser.add_argument(
        "--data_filepath", default="data/pg300_ccd10", help="The filepath for the images files."
    )
    parser.add_argument(
        "--time_filepath", default="./loriallen_times.dat", help="The filepath for the time stamps file."
    )
    parser.add_argument(
        "--psf_filepath", default=None, help="The filepath for the psf values file."
    )
    parser.add_argument(
        "--short", default=False, action="store_true", help="Use a coarse grid for a fast search."
    )
    args = parser.parse_args()

    # Set up the file path information.
    im_filepath = args.data_filepath
    time_file = args.time_filepath
    psf_file = args.psf_filepath
    results_suffix = "science_validation"
    if args.short:
        results_suffix = "science_validation_short"

    # Test whether we are regenerating the goldens.
    if args.generate_goldens:
        # Check whether the goldens directory exists (and create it
        # if not) and check whether the golden files already exist
        # (and prompt the user if so).
        check_and_create_goldens_dir()
        if check_goldens_exist(results_suffix):
            print("*** WARNING (re)generating golden files.")
            print("This will overwrite previous golden files.")
            answer = input("Proceed [y/N]: ")
            if answer != "y" and answer != "Y":
                print("Canceling and exiting.")
                sys.exit()

        # Run the search code and save the results to goldens/
        perform_search(im_filepath, time_file, psf_file, "goldens", results_suffix, args.short)
    else:
        if not check_goldens_exist(results_suffix):
            print("ERROR: Golden files do not exist. Generate new goldens using " "'--generate_goldens'")
        else:
            with tempfile.TemporaryDirectory() as dir_name:
                dir_name = "tmp"
                print("Running diff test with data in %s/" % im_filepath)
                print("Time file: %s" % time_file)
                print("PSF file: %s" % psf_file)
                if args.short:
                    print("Using a reduced parameter search (approximate).")

                # Do the search.
                perform_search(im_filepath, time_file, psf_file, dir_name, results_suffix, args.short)

                # Compare the result files.
                goldens_file = "goldens/results_%s.txt" % results_suffix
                new_results_file = "%s/results_%s.txt" % (dir_name, results_suffix)
                print("Comparing %s and %s" % (goldens_file, new_results_file))
                success = compare_result_files(goldens_file, new_results_file)

                # Compare the PS files.
                if success:
                    goldens_file = "goldens/ps_%s.txt" % results_suffix
                    new_results_file = "%s/ps_%s.txt" % (dir_name, results_suffix)
                    print("Comparing %s and %s" % (goldens_file, new_results_file))
                    success = compare_ps_files(goldens_file, new_results_file)

                if success:
                    print("\n*****\nDiff test PASSED.")
                else:
                    print("\n*****\nDiff test FAILED.")
