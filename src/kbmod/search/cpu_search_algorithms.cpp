/*
 * cpu_search_algorithms.cpp
 *
 * A purely CPU version of the critical kernel functions. While the __host__ tag
 * should be able to compile for CPU we fail if we do not have the nvidia driver
 * and libraries installed.
 */

#include "cpu_search_algorithms.h"

namespace search {

/*
 * Evaluate the likelihood score (as computed with from the psi and phi values) for a single
 * given candidate trajectory. Modifies the trajectory in place to update the number of
 * observations, likelihood, and flux.
 *
 * Does not do sigma-G filtering.
 */
void evaluate_trajectory_cpu(PsiPhiArray& psi_phi, Trajectory& candidate) {
    const unsigned int num_times = psi_phi.get_num_times();
    std::vector<float> psi_array;
    std::vector<float> phi_array;
    float psi_sum = 0.0;
    float phi_sum = 0.0;

    // Reset the statistics for the candidate.
    candidate.obs_count = 0;
    candidate.lh = -1.0;
    candidate.flux = -1.0;

    // Loop over each image and sample the appropriate pixel
    int num_seen = 0;
    for (unsigned int i = 0; i < num_times; ++i) {
        // Predict the trajectory's position.
        double curr_time = psi_phi.read_time(i);
        int current_x = (int)(floor(candidate.x + candidate.vx * curr_time + 0.5f));
        int current_y = (int)(floor(candidate.y + candidate.vy * curr_time + 0.5f));

        // Get the Psi and Phi pixel values. Skip invalid values, such as those marked NaN or NO_DATA.
        PsiPhi pixel_vals = psi_phi.read_psi_phi(i, current_y, current_x);
        if (isfinite(pixel_vals.psi) && isfinite(pixel_vals.phi)) {
            psi_sum += pixel_vals.psi;
            phi_sum += pixel_vals.phi;
            psi_array.push_back(pixel_vals.psi);
            phi_array.push_back(pixel_vals.phi);
            num_seen += 1;
        }
    }
    // Set stats (avoiding divide by zero of sqrt of negative).
    candidate.obs_count = num_seen;
    candidate.lh = (phi_sum > 0) ? (psi_sum / sqrt(phi_sum)) : -1.0;
    candidate.flux = (phi_sum > 0) ? (psi_sum / phi_sum) : -1.0;
}

/*
 * Perform the core KBMOD search (without sigma-G filtering) on CPU using
 * a naive nested loop.
 *
 * TODO: Add threading to speed this up.
 *
 */
void search_cpu_only(PsiPhiArray& psi_phi_array, SearchParameters params, TrajectoryList& trj_to_search,
                     TrajectoryList& results) {
    // Allocate space for all of the results.
    uint64_t height = psi_phi_array.get_height();
    uint64_t width = psi_phi_array.get_width();
    uint64_t total_results = params.results_per_pixel * height * width;
    results.resize(total_results);

    // Allocate space for a single pixel's results. We process one pixel at a time.
    uint64_t num_candidates = trj_to_search.get_size();
    TrajectoryList pixel_res(num_candidates);

    // Test each pixel using a giant nested loop.
    uint64_t next_result = 0;
    for (int y_i = params.y_start_min; y_i < params.y_start_max; ++y_i) {
        for (int x_i = params.x_start_min; x_i < params.x_start_max; ++x_i) {
            // Evaluate all the candidates.
            for (uint64_t trj_idx = 0; trj_idx < num_candidates; ++trj_idx) {
                Trajectory& candidate = trj_to_search.get_trajectory(trj_idx);
                Trajectory& curr_trj = pixel_res.get_trajectory(trj_idx);
                curr_trj.x = x_i;
                curr_trj.y = y_i;
                curr_trj.vx = candidate.vx;
                curr_trj.vy = candidate.vy;

                evaluate_trajectory_cpu(psi_phi_array, curr_trj);
            }

            // Sort the trajectories and save the best ones.
            pixel_res.sort_by_likelihood();
            for (int i = 0; i < params.results_per_pixel; ++i) {
                results.set_trajectory(next_result, pixel_res.get_trajectory(i));
                ++next_result;
            }
        }
    }
}

// Include CPU search algorithm bindings for testing.
#ifdef Py_PYTHON_H
static void cpu_search_algorithms_bindings(py::module& m) {
    m.def("evaluate_trajectory_cpu", &search::evaluate_trajectory_cpu);
    m.def("search_cpu_only", &search::search_cpu_only);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
