import os
import numpy as np
import sys
import argparse
from kbmod.run_search import run_search
from pathlib import Path

def check_and_create_results_dir(pg_num, ccd_name):
    """
    Test whether the results directory exists and create it if not.
    """
    dir_path = Path("/astro/users/jkubica/deep_res")
    if not dir_path.is_dir():
        os.mkdir("/astro/users/jkubica/deep_res")

    dir_path = Path(f"/astro/users/jkubica/deep_res/{pg_num}")
    if not dir_path.is_dir():
        os.mkdir(f"/astro/users/jkubica/deep_res/{pg_num}")
        
    dir_path = Path(f"/astro/users/jkubica/deep_res/{pg_num}/{ccd_name}")
    if not dir_path.is_dir():
        os.mkdir(f"/astro/users/jkubica/deep_res/{pg_num}/{ccd_name}")
        
if __name__ == "__main__":
    ccd_name = sys.argv[1]
    pg_num = sys.argv[2]
    im_filepath=(f"/epyc/projects3/smotherh/DEEP/warps/{pg_num}/{ccd_name}")
    res_filepath=(f"/astro/users/jkubica/deep_res/{pg_num}/{ccd_name}")
    time_file=(
        "/epyc/projects3/smotherh/DEEP/times/{}_times.dat".format(pg_num))
    results_suffix = "FAKE_DEEP"

    check_and_create_results_dir(pg_num, ccd_name)
    
    v_min = 90. # Pixels/day
    v_max = 400.
    #Offset by PI for prograde orbits in lori allen data
    ang_below = -np.pi+np.radians(45) # Angle below ecliptic
    ang_above = np.pi+np.radians(45) # Angle above ecliptic
    v_steps = 50
    ang_steps = 30

    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]
    #v_arr = [255,275,100]
    #ang_arr = [np.pi/15,np.pi/15,50]
    num_obs = 50
    if int(pg_num[4:8]) > 2020:
        visit_id_format = '{0:07d}.fits'
        visit_in_filename=[0,7]
    else:
        visit_id_format = '{0:06d}.fits'
        visit_in_filename=[0,6]

    input_parameters = {
        'im_filepath':im_filepath, 'res_filepath':res_filepath,
        'time_file':time_file, 'output_suffix':results_suffix, 'v_arr':v_arr,
        'ang_arr':ang_arr, 'num_obs':num_obs, 'do_mask':True, 'lh_level':7.,
        'sigmaG_lims':[25,75], 'mom_lims':[37.5,37.5,1.5,1.0,1.0],
        'peak_offset':[3.0,3.0], 'chunk_size':1000000, 'stamp_type':'parallel_sum',
        'eps':0.0025, 'gpu_filter':True, 'clip_negative':True,
        'sigmaG_filter_type':'both', 'cluster_type':'mid_position', 'mask_num_images':70,
        'visit_in_filename':visit_in_filename
    }
    rs = run_search(input_parameters)

    rs.run_search()
