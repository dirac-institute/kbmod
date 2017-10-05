from kbmodpy import kbmod as kb
import rawSearch

parameters = {
    "path": '../../../HITS/test_35/4,6tempExp/new_header/',
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
    "real_results": [],
    "flags": ~0,
    "flag_exceptions": [32,39],
    "master_flags": int('100111', 2),
    "master_threshold": 2,
    "save_results": True,
    "results_file_path": './results/',
    "results_count": 1800000,
    "cluster_eps": 0.004,
    "match_v": 0.02,
    "match_coord": 1,
    "save_science": True,
    "save_psi": False,
    "save_phi": False,
    "img_save_path": './images/'
}

total_matched = []
total_unmatched = []
all_stamps = []

mf = open("key.txt", "a")
mf.write(str(parameters)+'\n::::::\n')
mf.close()

runs = 5000

for i in range(runs):
    print("running search iteration " + str(i)+ '\n')
    key = rawSearch.run_search(parameters, i)
    with open("key.txt", "a") as matched_file:
        for k in key:
            matched_file.write(str(k)+'\n')
