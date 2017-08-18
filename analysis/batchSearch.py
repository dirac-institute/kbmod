from kbmodpy import kbmod as kb
import fullSearch

real_result = kb.trajectory()
real_result.flux = 5300
real_result.x = 3123
real_result.y = 3043
real_result.x_v = 2425
real_result.y_v = 1050

parameters = {
    "path": '../../HITS/test_35/4,6tempExp/new_header/',
    "max_images": 5,
    "psf_width": 1.5,
    "object_count": 4,
    "x_range": (5,3650),
    "y_range":(5, 3650),
    "xv_range": (1800,2500),
    "yv_range": (600,1400),
    "flux_range": (8999, 9000),
    "min_observations": 3,
    "angle_steps": 130,
    "velocity_steps": 80,
    "real_results": [real_result],
    "flags": ~0,
    "flag_exceptions": [32,39],
    "master_flags": int('100111', 2),
    "master_threshold": 2,
    "results_count": 150000,
    "cluster_eps": 0.004,
    "match_v": 0.025,
    "match_coord": 2
}

total_matched = []
total_unmatched = []
all_stamps = []

runs = 1000

for i in range(runs):
    print("running search iteration " + str(i)+ '\n')
    r_m, r_um, stmp = fullSearch.run_search(parameters)
    with open("matched.txt", "a") as matched_file:
        matched_file.write(str(r_m)+'  \n  ')
    with open("unmatched.txt", "a") as unmatched_file:
        unmatched_file.write(str(r_um)+'  \n  ')
