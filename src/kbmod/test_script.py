import kbmod

from kbmod.results import aggregate_rates_from_directories, compute_trajectory_rates

"""
This uses KBMOD branch result_trajectories
"""

#single_res = kbmod.results.Results.read_table(
#    "/sdf/scratch/rubin/kbmod/runs/20260312/5.2_20X20/large_piles/output/263242_5.2_20X20_302_to_401.collection.wu.5.2.repro.search.parquet",
#)

#flattened_table = compute_trajectory_rates(single_res)

# Generate a compbined file from all results files by recursively searching the top_dir
#top_dir = "/sdf/scratch/rubin/kbmod/runs/01202026/65.0_20X20/large_piles/0_to_99/search_results/search_468929_65.0_20X20_0_to_99_20260121_134357"
#output_filename = f"{top_dir}/result_rates_pre_nh_468929.parquet"

top_dir = "/sdf/scratch/rubin/kbmod/runs/20260326/"
output_filename = f"{top_dir}/result_rates_v3.parquet"

aggregate_rates_from_directories(
    top_dir,
    output_filename,
    chunk_size=10000,
    default_pixel_scale=5.5555555555556e-05, # Degrees per pixel
    requested_columns=["global_ra", "global_dec", "img_ra", "img_dec", "obs_valid"],
)