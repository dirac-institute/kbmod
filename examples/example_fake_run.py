import numpy as np
import argparse
import os
from fake_search import run_search

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', dest='im_filepath')
    parser.add_argument('--results_dir', dest='res_filepath')
    parser.add_argument('--results_suffix',
                        dest='results_suffix')
    parser.add_argument('--time_file', dest='time_file')

    args = parser.parse_args()
    im_filepath = args.im_filepath
    res_filepath = args.res_filepath
    results_suffix = args.results_suffix
    time_file = args.time_file

    patch_list = os.listdir(im_filepath)
    print(patch_list)
    
    v_min = 92. # Pixels/day
    v_max = 526.
    ang_below = np.pi/15. # Angle below ecliptic
    ang_above = np.pi/15. # Angle above ecliptic

    v_steps = 250 # Number of steps in velocity
    ang_steps = 128 # Number of steps in angle

    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]

    num_obs = 6

    rs = run_search(v_arr, ang_arr, num_obs)

    for patch_dir in patch_list:
        patch_str_arr = np.array([x for x in patch_dir])
        rand_seed = int(patch_str_arr[0])*10 + int(patch_str_arr[2])
        print(patch_dir, rand_seed)
        rs.run_search(os.path.join(im_filepath, patch_dir),
                      res_filepath, patch_dir,
                      time_file, mjd_lims=[57070.0, 57072.9],
                      rand_seed=rand_seed)
