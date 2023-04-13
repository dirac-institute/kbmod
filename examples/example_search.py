import argparse

import numpy as np

from kbmod.run_search import run_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", dest="im_filepath")
    parser.add_argument("--results_dir", dest="res_filepath")
    parser.add_argument("--results_suffix", dest="results_suffix")
    parser.add_argument("--time_file", dest="time_file")

    args = parser.parse_args()
    im_filepath = args.im_filepath
    res_filepath = args.res_filepath
    results_suffix = args.results_suffix
    time_file = args.time_file

    v_min = 92.0  # Pixels/day
    v_max = 526.0
    ang_below = -np.pi + np.pi / 15.0  # Angle below ecliptic
    ang_above = np.pi + np.pi / 15.0  # Angle above ecliptic

    # Use a reduced number of steps to run quickly
    v_steps = 51  # 512 for fuller search
    ang_steps = 25  # 265 for fuller search

    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]
    num_obs = 16

    input_parameters = {
        "im_filepath": im_filepath,
        "res_filepath": res_filepath,
        "time_file": time_file,
        "output_suffix": results_suffix,
        "v_arr": v_arr,
        "ang_arr": ang_arr,
        "num_obs": num_obs,
        "do_mask": False,
    }
    rs = run_search(input_parameters)

    rs.run_search()
