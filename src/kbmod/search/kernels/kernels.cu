/*
 * kernels.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef KERNELS_CU_
#define KERNELS_CU_

#include <assert.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <stdio.h>
#include <float.h>

#include "../common.h"
#include "../gpu_array.h"
#include "../psi_phi_array_ds.h"
#include "../trajectory_list.h"

#include "kernel_memory.h"

namespace search {

// ---------------------------------------
// --- Data Access Functions -------------
// ---------------------------------------

__host__ __device__ bool device_pixel_valid(float value) { return isfinite(value); }

__host__ __device__ int predict_index(float pos0, float vel0, double time) {
    return (int)(floor(pos0 + vel0 * time + 0.5f));
}

__host__ __device__ PsiPhi read_encoded_psi_phi(PsiPhiArrayMeta &params, void *psi_phi_vect, uint64_t time,
                                                int row, int col) {
    if ((row < 0) || (col < 0) || (row >= params.height) || (col >= params.width) ||
        (time >= params.num_times) || (psi_phi_vect == nullptr)) {
        return {NO_DATA, NO_DATA};
    }
    uint64_t u_row = static_cast<uint64_t>(row);
    uint64_t u_col = static_cast<uint64_t>(col);

    // Compute the in-list index from the row, column, and time.
    uint64_t start_index = 2 * (params.pixels_per_image * time + u_row * params.width + u_col);
    if (start_index >= params.num_entries - 1) {
        return {NO_DATA, NO_DATA};
    }

    if (params.num_bytes == 4) {
        // Short circuit the typical case of float encoding. No scaling or shifting done.
        return {reinterpret_cast<float *>(psi_phi_vect)[start_index],
                reinterpret_cast<float *>(psi_phi_vect)[start_index + 1]};
    }

    // Handle the compressed encodings.
    PsiPhi result;
    float psi_value = (params.num_bytes == 1)
                              ? (float)reinterpret_cast<uint8_t *>(psi_phi_vect)[start_index]
                              : (float)reinterpret_cast<uint16_t *>(psi_phi_vect)[start_index];
    result.psi = (psi_value == 0.0) ? NO_DATA : (psi_value - 1.0) * params.psi_scale + params.psi_min_val;

    float phi_value = (params.num_bytes == 1)
                              ? (float)reinterpret_cast<uint8_t *>(psi_phi_vect)[start_index + 1]
                              : (float)reinterpret_cast<uint16_t *>(psi_phi_vect)[start_index + 1];
    result.phi = (phi_value == 0.0) ? NO_DATA : (phi_value - 1.0) * params.phi_scale + params.phi_min_val;

    return result;
}

// ---------------------------------------
// --- Computation Functions -------------
// ---------------------------------------

extern "C" __device__ __host__ void SigmaGFilteredIndicesCU(float *values, int num_values, float sgl0,
                                                            float sgl1, float sigmag_coeff, float width,
                                                            int *idx_array, int *min_keep_idx,
                                                            int *max_keep_idx) {
    // Basic data checking. We don't use assert here because assert does not work in __device__ functions.
    // So we ignore the error and return so we do not access invalid memory.
    if ((idx_array == nullptr) || (min_keep_idx == nullptr) && (max_keep_idx == nullptr)) {
        return;
    }
    if (num_values == 0) {
        // Exit early if there are no values.
        *min_keep_idx = 0;
        *max_keep_idx = -1;
        return;
    }

    // Clip the percentiles to [0.01, 99.99] to avoid invalid array accesses.
    if (sgl0 < 0.0001) sgl0 = 0.0001;
    if (sgl1 > 0.9999) sgl1 = 0.9999;

    // Initialize the index array.
    for (int j = 0; j < num_values; j++) {
        idx_array[j] = j;
    }

    // Sort the the indexes (idx_array) of values in ascending order.
    int tmp_sort_idx;
    for (int j = 0; j < num_values; j++) {
        for (int k = j + 1; k < num_values; k++) {
            if (values[idx_array[j]] > values[idx_array[k]]) {
                tmp_sort_idx = idx_array[j];
                idx_array[j] = idx_array[k];
                idx_array[k] = tmp_sort_idx;
            }
        }
    }

    // Compute the index of each of the percent values in values
    // from the given bounds sgl0, 0.5 (median), and sgl1.
    // Make sure all indices are valid (in [0, num_values-1]).
    int pct_L = int(ceil(num_values * sgl0) + 0.001) - 1;
    pct_L = (pct_L < 0) ? 0 : pct_L;
    pct_L = (pct_L >= num_values) ? (num_values - 1) : pct_L;

    int pct_H = int(ceil(num_values * sgl1) + 0.001) - 1;
    pct_H = (pct_H < 0) ? 0 : pct_H;
    pct_H = (pct_H >= num_values) ? (num_values - 1) : pct_H;

    int median_ind = int(ceil(num_values * 0.5) + 0.001) - 1;
    median_ind = (median_ind < 0) ? 0 : median_ind;
    median_ind = (median_ind >= num_values) ? (num_values - 1) : median_ind;

    // Compute the values that are +/- (width * sigma_g) from the median.
    float sigma_g = sigmag_coeff * (values[idx_array[pct_H]] - values[idx_array[pct_L]]);
    float min_value = values[idx_array[median_ind]] - width * sigma_g;
    float max_value = values[idx_array[median_ind]] + width * sigma_g;

    // Find the index of the first value >= min_value.
    int start = 0;
    while ((start < median_ind) && (values[idx_array[start]] < min_value)) {
        ++start;
    }
    *min_keep_idx = start;

    // Find the index of the last value <= max_value.
    int end = median_ind + 1;
    while ((end < num_values) && (values[idx_array[end]] <= max_value)) {
        ++end;
    }
    *max_keep_idx = end - 1;
}

/*
 * Evaluate the likelihood score (as computed with from the psi and phi values) for a single
 * given candidate trajectory. Modifies the trajectory in place to update the number of
 * observations, likelihood, and flux.
 */
extern "C" __device__ __host__ void evaluateTrajectory(PsiPhiArrayMeta psi_phi_meta, void *psi_phi_vect,
                                                       double *image_times, SearchParameters params,
                                                       Trajectory *candidate) {
    // Basic data checking. We don't use assert here because assert does not work in __device__ functions.
    // So we ignore the error and return so we do not access invalid memory.
    if ((psi_phi_vect == nullptr) || (image_times == nullptr) || (candidate == nullptr)) return;
    if (psi_phi_meta.num_times >= MAX_NUM_IMAGES) return;

    // Data structures used for filtering. We fill in only what we need.
    float psi_array[MAX_NUM_IMAGES];
    float phi_array[MAX_NUM_IMAGES];
    float psi_sum = 0.0;
    float phi_sum = 0.0;

    // Reset the statistics for the candidate.
    candidate->obs_count = 0;
    candidate->lh = -1.0;
    candidate->flux = -1.0;

    // Loop over each image and sample the appropriate pixel
    int num_seen = 0;
    for (unsigned int i = 0; i < psi_phi_meta.num_times; ++i) {
        // Predict the trajectory's position.
        double curr_time = image_times[i];
        int current_x = predict_index(candidate->x, candidate->vx, curr_time);
        int current_y = predict_index(candidate->y, candidate->vy, curr_time);

        // Get the Psi and Phi pixel values. Skip invalid values, such as those marked NaN or NO_DATA.
        PsiPhi pixel_vals = read_encoded_psi_phi(psi_phi_meta, psi_phi_vect, i, current_y, current_x);
        if (device_pixel_valid(pixel_vals.psi) && device_pixel_valid(pixel_vals.phi)) {
            psi_sum += pixel_vals.psi;
            phi_sum += pixel_vals.phi;
            psi_array[num_seen] = pixel_vals.psi;
            phi_array[num_seen] = pixel_vals.phi;
            num_seen += 1;

            // This should not be triggered due to prior checks, but just in case.
            if (num_seen >= MAX_NUM_IMAGES) break;
        }
    }
    // Set stats (avoiding divide by zero of sqrt of negative).
    candidate->obs_count = num_seen;
    candidate->lh = (phi_sum > 0) ? (psi_sum / sqrt(phi_sum)) : -1.0;
    candidate->flux = (phi_sum > 0) ? (psi_sum / phi_sum) : -1.0;

    // If we do not have enough observations or a good enough LH score,
    // do not bother with any of the following steps.
    if ((candidate->obs_count < params.min_observations) || (candidate->obs_count == 0) ||
        (params.do_sigmag_filter && candidate->lh < params.min_lh))
        return;

    // Safety check to avoid out of bounds memory access.
    if (num_seen > MAX_NUM_IMAGES) {
        // Unfortunately we cannot raise an error in a kernel, so we print to stdout and exit.
        printf("ERROR: num_seen=%d exceeds MAX_NUM_IMAGES=%d in evaluateTrajectory.\n", num_seen, MAX_NUM_IMAGES);
        return;
    }

    // If we are doing on GPU filtering, run the sigma_g filter and recompute the likelihoods.
    if (params.do_sigmag_filter) {
        // Fill in a likelihood and index array for sorting.
        float lc_array[MAX_NUM_IMAGES];
        int idx_array[MAX_NUM_IMAGES];
        for (int i = 0; i < num_seen; ++i) {
            lc_array[i] = (phi_array[i] != 0) ? (psi_array[i] / phi_array[i]) : 0;
            idx_array[i] = i;
        }

        int min_keep_idx = 0;
        int max_keep_idx = num_seen - 1;
        SigmaGFilteredIndicesCU(lc_array, num_seen, params.sgl_L, params.sgl_H, params.sigmag_coeff, 2.0,
                                idx_array, &min_keep_idx, &max_keep_idx);
        if (min_keep_idx < 0) min_keep_idx = 0;
        if (max_keep_idx >= num_seen) max_keep_idx = num_seen - 1;

        // Compute the likelihood and flux of the track based on the filtered
        // observations (ones in [min_keep_idx, max_keep_idx]).
        float new_psi_sum = 0.0;
        float new_phi_sum = 0.0;
        for (int i = min_keep_idx; i <= max_keep_idx; i++) {
            int idx = idx_array[i];
            new_psi_sum += psi_array[idx];
            new_phi_sum += phi_array[idx];
        }
        // Set likelihood and flux (avoiding divide by zero of sqrt of negative).
        candidate->lh = (new_phi_sum > 0) ? (new_psi_sum / sqrt(new_phi_sum)) : -1.0;
        candidate->flux = (new_phi_sum > 0) ? (new_psi_sum / new_phi_sum) : -1.0;
    }
}

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Returns a
 * fixed number of results per pixel specified by params.results_per_pixel.
 * Filters the results using a sigma_g-based filter and a central-moment filter.
 *
 * Creates a local copy of psi_phi_meta and params in local memory space.
 */
__global__ void searchFilterImages(PsiPhiArrayMeta psi_phi_meta, void *psi_phi_vect, double *image_times,
                                   SearchParameters params, uint64_t num_trajectories,
                                   Trajectory *trajectories, Trajectory *results) {
    // Basic data validity check.
    assert(psi_phi_vect != nullptr && image_times != nullptr && trajectories != nullptr &&
           results != nullptr);

    // copy their time before progressing. We need to do this before pruning on
    // (x, y) in order to correctly handle blocks at the edge of the image.
    __shared__ double shared_times[MAX_NUM_IMAGES];
    int time_idx = threadIdx.x + threadIdx.y * blockDim.x;
    if ((time_idx < psi_phi_meta.num_times) && (time_idx < MAX_NUM_IMAGES)) {
        shared_times[time_idx] = image_times[time_idx];
    }
    __syncthreads();  // Block until all are done loading.

    // Get the x and y coordinates within the search space.
    const int x_i = blockIdx.x * THREAD_DIM_X + threadIdx.x;
    const int y_i = blockIdx.y * THREAD_DIM_Y + threadIdx.y;

    // Check that the x and y coordinates are consistent with the search space.
    const int search_width = params.x_start_max - params.x_start_min;
    const int search_height = params.y_start_max - params.y_start_min;
    if ((x_i < 0) || (y_i < 0) || (x_i >= search_width) || (y_i >= search_height)) {
        return;
    }

    // Get origin pixel for the trajectories in pixel space.
    const int x = x_i + params.x_start_min;
    const int y = y_i + params.y_start_min;

    // Create an initial set of best results with likelihood -1.0 and default
    // values for everything so that we do not propogate uninitialized values.
    const uint64_t base_index = (y_i * search_width + x_i) * params.results_per_pixel;
    if (base_index + params.results_per_pixel > params.total_results) {
        // Unfortunately we cannot raise an error in a kernel, so we print to stdout and exit.
        printf("ERROR: base_index=%llu out of bounds in searchFilterImages kernel.\n", base_index);
        return;
    }

    for (int r = 0; r < params.results_per_pixel; ++r) {
        results[base_index + r].x = x;
        results[base_index + r].y = y;
        results[base_index + r].vx = 0.0f;
        results[base_index + r].vy = 0.0f;
        results[base_index + r].lh = -FLT_MAX;
        results[base_index + r].flux = 0.0f;
        results[base_index + r].obs_count = 0;
    }

    // For each trajectory we'd like to search
    for (uint64_t t = 0; t < num_trajectories; ++t) {
        // Create a trajectory for this search.
        Trajectory curr_trj;
        curr_trj.x = x;
        curr_trj.y = y;
        curr_trj.vx = trajectories[t].vx;
        curr_trj.vy = trajectories[t].vy;
        curr_trj.obs_count = 0;

        // Evaluate the trajectory.
        evaluateTrajectory(psi_phi_meta, psi_phi_vect, shared_times, params, &curr_trj);

        // If we do not have enough observations or a good enough LH score,
        // do not bother inserting it into the sorted list of results.
        if ((curr_trj.obs_count < params.min_observations) ||
            (params.do_sigmag_filter && curr_trj.lh < params.min_lh))
            continue;

        // Insert the new trajectory into the sorted list of final results.
        Trajectory temp;
        for (unsigned int r = 0; r < params.results_per_pixel; ++r) {
            if (curr_trj.lh > results[base_index + r].lh) {
                temp = results[base_index + r];
                results[base_index + r] = curr_trj;
                curr_trj = temp;
            }
        }
    }
}

extern "C" void deviceSearchFilter(PsiPhiArray &psi_phi_array, SearchParameters params,
                                   TrajectoryList &trj_to_search, TrajectoryList &results) {
    // Check the hard coded maximum number of images against the num_images.
    uint64_t num_images = psi_phi_array.get_num_times();
    if (num_images > MAX_NUM_IMAGES) {
        throw std::runtime_error("Number of images exceeds GPU maximum " + std::to_string(MAX_NUM_IMAGES));
    }
    if (THREAD_DIM_X * THREAD_DIM_Y < MAX_NUM_IMAGES) {
        throw std::runtime_error("Insufficient threads to load all the times.");
    }

    // Check that the device vectors have already been allocated.
    if (!psi_phi_array.on_gpu()) {
        throw std::runtime_error("PsiPhi data is not on GPU.");
    }
    if (psi_phi_array.get_gpu_array_ptr() == nullptr) {
        throw std::runtime_error("PsiPhi data has not been created.");
    }
    if (psi_phi_array.get_gpu_time_array_ptr() == nullptr) {
        throw std::runtime_error("GPU time data has not been created.");
    }

    // Make sure the trajectory data is allocated on the GPU.
    if (!trj_to_search.on_gpu()) trj_to_search.move_to_gpu();
    uint64_t num_trajectories = trj_to_search.get_size();
    Trajectory *device_tests = trj_to_search.get_gpu_list_ptr();
    if (device_tests == nullptr) throw std::runtime_error("Invalid test list pointer.");

    if (!results.on_gpu()) results.move_to_gpu();
    uint64_t num_results = results.get_size();
    Trajectory *device_results = results.get_gpu_list_ptr();
    if (device_results == nullptr) throw std::runtime_error("Invalid result list pointer.");

    // Compute the range of starting pixels to use when setting the blocks and threads.
    // We use the width and height of the search space (as opposed to the image width
    // and height), meaning the blocks/threads will be indexed relative to the search space.
    int search_width = params.x_start_max - params.x_start_min;
    int search_height = params.y_start_max - params.y_start_min;
    if ((search_width <= 0) || (search_height <= 0))
        throw std::runtime_error("Invalid search bounds x=[" + std::to_string(params.x_start_min) + ", " +
                                 std::to_string(params.x_start_max) + "] y=[" +
                                 std::to_string(params.y_start_min) + ", " +
                                 std::to_string(params.y_start_max) + "]");
                                 
    // Check that we have enough result space allocated. num_results is the number of spaces
    // we have in the results vector and expected_results is the number we are going to
    // generate. So we need num_results >= expected_results to have enough storage space.
    uint64_t expected_results = params.results_per_pixel * search_width * search_height;
    params.total_results = expected_results;
    if (num_results < expected_results) {
        throw std::runtime_error("Not enough space allocated for results. Requires: " +
                                 std::to_string(expected_results) + ". Received: " +
                                 std::to_string(num_results));
    }

    dim3 blocks(search_width / THREAD_DIM_X + 1, search_height / THREAD_DIM_Y + 1);
    dim3 threads(THREAD_DIM_X, THREAD_DIM_Y);

    // Launch Search
    searchFilterImages<<<blocks, threads>>>(psi_phi_array.get_meta_data(), psi_phi_array.get_gpu_array_ptr(),
                                            psi_phi_array.get_gpu_time_array_ptr(), params, num_trajectories,
                                            device_tests, device_results);
    cudaDeviceSynchronize();
}

} /* namespace search */

#endif /* KERNELS_CU_ */
