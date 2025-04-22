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

} /* namespace search */
