import argparse

import numpy as np
from run_search import region_search

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

    v_x = 214.0  # Pixels/day
    v_y = 81.0

    v_arr = [v_x, v_y]
    radius = 25

    num_obs = 6

    rs = region_search(v_arr, radius, num_obs)

    rs.run_search(
        im_filepath,
        res_filepath,
        results_suffix,
        time_file,
        likelihood_level=19.0,
        mjd_lims=[57070.0, 57072.9],
    )
