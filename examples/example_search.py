import numpy as np
import argparse
from run_search import run_search

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
    
    v_min = 92. # Pixels/day
    v_max = 526.
    #v_min=250
    #v_max=280
    ang_below = -np.pi+np.pi/15. # Angle below ecliptic
    ang_above = np.pi+np.pi/15. # Angle above ecliptic
    #ang_below=np.pi/15
    #ang_above=np.pi/15
    v_steps = 256 # Number of steps in velocity
    ang_steps = 128 # Number of steps in angle

    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]

    num_obs = 6

    rs = run_search(v_arr, ang_arr, num_obs)

    rs.run_search(im_filepath, res_filepath, results_suffix,
                  time_file,lh_level=10.0)
