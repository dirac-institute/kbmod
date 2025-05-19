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
void evaluate_trajectory_cpu(const PsiPhiArray& psi_phi, Trajectory& candidate) {
    const unsigned int num_times = psi_phi.get_num_times();
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
            num_seen += 1;
        }
    }
    // Set stats (avoiding divide by zero of sqrt of negative).
    candidate.obs_count = num_seen;
    candidate.lh = (phi_sum > 0) ? (psi_sum / sqrt(phi_sum)) : -1.0;
    candidate.flux = (phi_sum > 0) ? (psi_sum / phi_sum) : -1.0;
}

/*
 * Evaluate all of the candidate trajectories from a single starting pixel
 * (y, x) and return the best "num_results".
 *
 */
std::vector<Trajectory> evaluate_single_pixel(int y, int x, const PsiPhiArray& psi_phi,
                                              TrajectoryList& trj_to_search, int num_results) {
    // Allocate space for this search.
    uint64_t num_candidates = trj_to_search.get_size();
    TrajectoryList pixel_res(num_candidates);

    // Evaluate all of the candidate trajectories for this pixel.
    for (uint64_t trj_idx = 0; trj_idx < num_candidates; ++trj_idx) {
        Trajectory& candidate = trj_to_search.get_trajectory(trj_idx);

        Trajectory& curr_trj = pixel_res.get_trajectory(trj_idx);
        curr_trj.x = x;
        curr_trj.y = y;
        curr_trj.vx = candidate.vx;
        curr_trj.vy = candidate.vy;
        curr_trj.flux = 0.0;
        curr_trj.obs_count = 0.0;

        evaluate_trajectory_cpu(psi_phi, curr_trj);
    }

    // Sort the trajectories and save the best ones.
    pixel_res.sort_by_likelihood();
    return pixel_res.get_batch(0, num_results);
}

/*
 * Perform the core KBMOD search (without sigma-G filtering) on CPU using
 * a naive nested loop.
 *
 */
void search_cpu_only(PsiPhiArray& psi_phi_array, SearchParameters params, TrajectoryList& trj_to_search,
                     TrajectoryList& results) {
    // Allocate space for all of the results.
    uint64_t search_height = params.y_start_max - params.y_start_min;
    uint64_t search_width = params.x_start_max - params.x_start_min;
    uint64_t total_results = params.results_per_pixel * search_height * search_width;
    results.resize(total_results);

// Test each pixel using a giant nested loop.  Allow omp to dynamically
// thread the computations.
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y_i = 0; y_i < search_height; ++y_i) {
        for (int x_i = 0; x_i < search_width; ++x_i) {
            std::vector<Trajectory> pixel_res =
                    evaluate_single_pixel(y_i + params.y_start_min, x_i + params.x_start_min, psi_phi_array,
                                          trj_to_search, params.results_per_pixel);

// We restrict the writing of results to a single thread.  The batch of results
// is inserted into a specific location within the full results list.
#pragma omp critical
            {
                uint64_t start_ind = (y_i * search_width + x_i) * params.results_per_pixel;
                for (uint64_t i = 0; i < params.results_per_pixel; ++i) {
                    results.set_trajectory(start_ind + i, pixel_res[i]);
                }
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
