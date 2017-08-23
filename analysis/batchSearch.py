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
    "angle_range": (0.2,0.5),
    "velocity_range": (2100,2800),
    "flux_range": (250, 7000),
    "min_observations": 3,
    "angle_steps": 170,
    "velocity_steps": 100,
    "search_margin": 1.05,
    "real_results": [real_result],
    "flags": ~0,
    "flag_exceptions": [32,39],
    "master_flags": int('100111', 2),
    "master_threshold": 2,
    "results_count": 1800000,
    "cluster_eps": 0.004,
    "match_v": 0.02,
    "match_coord": 1
}

total_matched = []
total_unmatched = []
all_stamps = []

mf = open("matched.txt", "a")
mf.write(str(parameters)+'\n::::::\n')
mf.close()
umf = open("unmatched.txt", "a")
umf.write(str(parameters)+'\n::::::\n')
umf.close()

runs = 5000

for i in range(runs):
    print("running search iteration " + str(i)+ '\n')
    r_m, r_um, stmp = fullSearch.run_search(parameters)
    with open("matched.txt", "a") as matched_file:
        for res in r_m:
            matched_file.write(str(res)+'\n')
    with open("unmatched.txt", "a") as unmatched_file:
        for res in r_um:
            unmatched_file.write(str(res)+'\n')
