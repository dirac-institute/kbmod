/*
 * kernels.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef KERNELS_CU_
#define KERNELS_CU_
#define GPU_LC_FILTER 1
#define MAX_NUM_IMAGES 140
#define MAX_STAMP_IMAGES 200

#include <cmath>
#include <stdexcept>
#include <vector>
#include <stdio.h>
#include <float.h>

#include "common.h"
#include "cuda_errors.h"
#include "psi_phi_array_ds.h"

namespace search {

extern "C" void device_allocate_psi_phi_array(PsiPhiArray* data) {
    if (!data->cpu_array_allocated())
        throw std::runtime_error("CPU data is not allocated.");
    if (data->gpu_array_allocated())
        throw std::runtime_error("GPU data is already allocated.");

    void* device_array_ptr;
    checkCudaErrors(cudaMalloc((void **)&device_array_ptr, data->get_total_array_size()));
    checkCudaErrors(cudaMemcpy(device_array_ptr,
                               data->get_cpu_array_ptr(),
                               data->get_total_array_size(),
                               cudaMemcpyHostToDevice));
    data->set_gpu_array_ptr(device_array_ptr);
}

extern "C" void device_free_psi_phi_array(PsiPhiArray* data) {
    if (data->gpu_array_allocated()) {
        checkCudaErrors(cudaFree(data->get_gpu_array_ptr()));
        data->set_gpu_array_ptr(nullptr);
    }
}

__forceinline__ __device__ PsiPhi read_encoded_psi_phi(PsiPhiArrayMeta &params, void *psi_phi_vect, int time,
                                                       int row, int col) {
    // Bounds checking.
    if ((row < 0) || (col < 0) || (row >= params.height) || (col >= params.width)) {
        return {NO_DATA, NO_DATA};
    }

    // Compute the in-list index from the row, column, and time.
    int start_index = 2 * (params.pixels_per_image * time + row * params.width + col);
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

extern "C" __device__ __host__ void SigmaGFilteredIndicesCU(float *values, int num_values, float sgl0,
                                                            float sgl1, float sigmag_coeff, float width,
                                                            int *idx_array, int *min_keep_idx,
                                                            int *max_keep_idx) {
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
    const int pct_L = int(ceil(num_values * sgl0) + 0.001) - 1;
    const int pct_H = int(ceil(num_values * sgl1) + 0.001) - 1;
    const int median_ind = int(ceil(num_values * 0.5) + 0.001) - 1;

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
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Returns a
 * fixed number of results per pixel specified by RESULTS_PER_PIXEL
 * filters results using a sigma_g-based filter and a central-moment filter.
 *
 * Creates a local copy of psi_phi_meta and params in local memory space.
 */
__global__ void searchFilterImages(PsiPhiArrayMeta psi_phi_meta, void *psi_phi_vect, float *image_times,
                                   SearchParameters params, int num_trajectories, Trajectory *trajectories,
                                   Trajectory *results) {
    // Get the x and y coordinates within the search space.
    const int x_i = blockIdx.x * THREAD_DIM_X + threadIdx.x;
    const int y_i = blockIdx.y * THREAD_DIM_Y + threadIdx.y;

    // Check that the x and y coordinates are consistent with the search space.
    const int search_width = params.x_start_max - params.x_start_min;
    const int search_height = params.y_start_max - params.y_start_min;
    if ((x_i >= search_width) || (y_i >= search_height)) {
        return;
    }

    // Get origin pixel for the trajectories in pixel space.
    const int x = x_i + params.x_start_min;
    const int y = y_i + params.y_start_min;

    // Data structures used for filtering.
    float lc_array[MAX_NUM_IMAGES];
    float psi_array[MAX_NUM_IMAGES];
    float phi_array[MAX_NUM_IMAGES];
    int idx_array[MAX_NUM_IMAGES];

    // Create an initial set of best results with likelihood -1.0.
    // We also set (x, y) because they are used in the later python
    // functions.
    Trajectory best[RESULTS_PER_PIXEL];
    for (int r = 0; r < RESULTS_PER_PIXEL; ++r) {
        best[r].x = x;
        best[r].y = y;
        best[r].lh = -1.0;
    }

    // For each trajectory we'd like to search
    for (int t = 0; t < num_trajectories; ++t) {
        // Create a trajectory for this search.
        Trajectory curr_trj;
        curr_trj.x = x;
        curr_trj.y = y;
        curr_trj.vx = trajectories[t].vx;
        curr_trj.vy = trajectories[t].vy;
        curr_trj.obs_count = 0;

        float psi_sum = 0.0;
        float phi_sum = 0.0;

        // Loop over each image and sample the appropriate pixel
        for (int i = 0; i < psi_phi_meta.num_times; ++i) {
            lc_array[i] = 0;
            psi_array[i] = 0;
            phi_array[i] = 0;
            idx_array[i] = i;
        }

        // Loop over each image and sample the appropriate pixel
        int num_seen = 0;
        for (int i = 0; i < psi_phi_meta.num_times; ++i) {
            // Predict the trajectory's position.
            float curr_time = image_times[i];
            int current_x = x + int(curr_trj.vx * curr_time + 0.5);
            int current_y = y + int(curr_trj.vy * curr_time + 0.5);

            // Get the Psi and Phi pixel values.
            PsiPhi pixel_vals = read_encoded_psi_phi(psi_phi_meta, psi_phi_vect, i, current_y, current_x);
            if (pixel_vals.psi != NO_DATA && pixel_vals.phi != NO_DATA) {
                curr_trj.obs_count++;
                psi_sum += pixel_vals.psi;
                phi_sum += pixel_vals.phi;
                psi_array[num_seen] = pixel_vals.psi;
                phi_array[num_seen] = pixel_vals.phi;
                if (pixel_vals.phi != 0.0) lc_array[num_seen] = pixel_vals.psi / pixel_vals.phi;
                num_seen += 1;
            }
        }
        curr_trj.lh = psi_sum / sqrt(phi_sum);
        curr_trj.flux = psi_sum / phi_sum;

        // If we do not have enough observations or a good enough LH score,
        // do not bother with any of the following steps.
        if ((curr_trj.obs_count < params.min_observations) ||
            (params.do_sigmag_filter && curr_trj.lh < params.min_lh))
            continue;

        // If we are doing on GPU filtering, run the sigma_g filter
        // and recompute the likelihoods.
        if (params.do_sigmag_filter) {
            int min_keep_idx = 0;
            int max_keep_idx = num_seen - 1;
            SigmaGFilteredIndicesCU(lc_array, num_seen, params.sgl_L, params.sgl_H, params.sigmag_coeff, 2.0,
                                    idx_array, &min_keep_idx, &max_keep_idx);

            // Compute the likelihood and flux of the track based on the filtered
            // observations (ones in [min_keep_idx, max_keep_idx]).
            float new_psi_sum = 0.0;
            float new_phi_sum = 0.0;
            for (int i = min_keep_idx; i <= max_keep_idx; i++) {
                int idx = idx_array[i];
                new_psi_sum += psi_array[idx];
                new_phi_sum += phi_array[idx];
            }
            curr_trj.lh = new_psi_sum / sqrt(new_phi_sum);
            curr_trj.flux = new_psi_sum / new_phi_sum;
        }

        // Insert the new trajectory into the sorted list of results.
        // Only sort the values with valid likelihoods.
        Trajectory temp;
        for (int r = 0; r < RESULTS_PER_PIXEL; ++r) {
            if (curr_trj.lh > best[r].lh && curr_trj.lh > -1.0) {
                temp = best[r];
                best[r] = curr_trj;
                curr_trj = temp;
            }
        }
    }

    // Copy the sorted list of best results for this pixel into
    // the correct location within the global results vector.
    // Note the results index is based on the pixel values in search
    // space (not image space).
    const int base_index = (y_i * search_width + x_i) * RESULTS_PER_PIXEL;
    for (int r = 0; r < RESULTS_PER_PIXEL; ++r) {
        results[base_index + r] = best[r];
    }
}

extern "C" void deviceSearchFilter(PsiPhiArray &psi_phi_array, float *image_times, SearchParameters params,
                                   int num_trajectories, Trajectory *trj_to_search, int num_results,
                                   Trajectory *best_results) {
    // Allocate Device memory
    Trajectory *device_tests;
    float *device_img_times;
    Trajectory *device_search_results;

    // Check the hard coded maximum number of images against the num_images.
    int num_images = psi_phi_array.get_num_times();
    if (num_images > MAX_NUM_IMAGES) {
        throw std::runtime_error("Number of images exceeds GPU maximum.");
    }

    // Check that the device psi_phi vector has been allocated.
    if (psi_phi_array.gpu_array_allocated() == false) {
        throw std::runtime_error("PsiPhi data has not been created.");
    }

    // Copy trajectories to search
    if (params.debug) {
        printf("Allocating %lu bytes testing grid with %i elements.\n", sizeof(Trajectory) * num_trajectories,
               num_trajectories);
    }
    checkCudaErrors(cudaMalloc((void **)&device_tests, sizeof(Trajectory) * num_trajectories));
    checkCudaErrors(cudaMemcpy(device_tests, trj_to_search, sizeof(Trajectory) * num_trajectories,
                               cudaMemcpyHostToDevice));

    // Copy the time vector.
    if (params.debug) {
        printf("Allocating %lu bytes for time data.\n", sizeof(float) * num_images);
    }
    checkCudaErrors(cudaMalloc((void **)&device_img_times, sizeof(float) * num_images));
    checkCudaErrors(
            cudaMemcpy(device_img_times, image_times, sizeof(float) * num_images, cudaMemcpyHostToDevice));

    // Allocate space for the results.
    if (params.debug) {
        printf("Allocating %lu bytes for %i results.\n", sizeof(Trajectory) * num_results, num_results);
    }
    checkCudaErrors(cudaMalloc((void **)&device_search_results, sizeof(Trajectory) * num_results));

    // Compute the range of starting pixels to use when setting the blocks and threads.
    // We use the width and height of the search space (as opposed to the image width
    // and height), meaning the blocks/threads will be indexed relative to the search space.
    int search_width = params.x_start_max - params.x_start_min;
    int search_height = params.y_start_max - params.y_start_min;
    dim3 blocks(search_width / THREAD_DIM_X + 1, search_height / THREAD_DIM_Y + 1);
    dim3 threads(THREAD_DIM_X, THREAD_DIM_Y);

    // Launch Search
    searchFilterImages<<<blocks, threads>>>(psi_phi_array.get_meta_data(), psi_phi_array.get_gpu_array_ptr(), device_img_times,
                                            params, num_trajectories, device_tests, device_search_results);

    // Read back results
    checkCudaErrors(cudaMemcpy(best_results, device_search_results, sizeof(Trajectory) * num_results,
                               cudaMemcpyDeviceToHost));

    // Free the on GPU memory.
    checkCudaErrors(cudaFree(device_search_results));
    checkCudaErrors(cudaFree(device_img_times));
    checkCudaErrors(cudaFree(device_tests));
}

__global__ void deviceGetCoaddStamp(int num_images, int width, int height, float *image_vect,
                                    float* image_times, int num_trajectories, Trajectory *trajectories,
                                    StampParameters params, int *use_index_vect, float *results) {
    // Get the trajectory that we are going to be using.
    const int trj_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (trj_index < 0 || trj_index >= num_trajectories) return;
    Trajectory trj = trajectories[trj_index];

    // Get the pixel coordinates within the stamp to use.
    const int stamp_width = 2 * params.radius + 1;
    const int stamp_x = threadIdx.y;
    if (stamp_x < 0 || stamp_x >= stamp_width) return;

    const int stamp_y = threadIdx.z;
    if (stamp_y < 0 || stamp_y >= stamp_width) return;

    // Compute the various offsets for the indices.
    int use_index_offset = num_images * trj_index;
    int trj_offset = trj_index * stamp_width * stamp_width;
    int pixel_index = stamp_width * stamp_y + stamp_x;

    // Allocate space for the coadd information.
    assertm(num_images < MAX_STAMP_IMAGES, "Number of images exceedes MAX_STAMP_IMAGES");
    float values[MAX_STAMP_IMAGES];

    // Loop over each image and compute the stamp.
    int num_values = 0;
    for (int t = 0; t < num_images; ++t) {
        // Skip entries marked 0 in the use_index_vect.
        if (use_index_vect != nullptr && use_index_vect[use_index_offset + t] == 0) {
            continue;
        }

        // Predict the trajectory's position.
        float curr_time = image_times[t];
        int current_x = int(trj.x + trj.vx * curr_time);
        int current_y = int(trj.y + trj.vy * curr_time);

        // Get the stamp and add it to the list of values.
        int img_x = current_x - params.radius + stamp_x;
        int img_y = current_y - params.radius + stamp_y;
        if ((img_x >= 0) && (img_x < width) && (img_y >= 0) && (img_y < height)) {
            int pixel_index = width * height * t + img_y * width + img_x;
            if (image_vect[pixel_index] != NO_DATA) {
                values[num_values] = image_vect[pixel_index];
                ++num_values;
            }
        }
    }

    // If there are no values, just return 0.
    if (num_values == 0) {
        results[trj_offset + pixel_index] = 0.0;
        return;
    }

    // Do the actual computation from the values.
    float result = 0.0;
    int median_ind = num_values / 2;  // Outside switch to avoid compiler warnings.
    switch (params.stamp_type) {
        case STAMP_MEDIAN:
            // Sort the values in ascending order.
            for (int j = 0; j < num_values; j++) {
                for (int k = j + 1; k < num_values; k++) {
                    if (values[j] > values[k]) {
                        float tmp = values[j];
                        values[j] = values[k];
                        values[k] = tmp;
                    }
                }
            }

            // Take the median value of the pixels with data.
            if (num_values % 2 == 0) {
                result = (values[median_ind] + values[median_ind - 1]) / 2.0;
            } else {
                result = values[median_ind];
            }
            break;
        case STAMP_SUM:
            for (int t = 0; t < num_values; ++t) {
                result += values[t];
            }
            break;
        case STAMP_MEAN:
            for (int t = 0; t < num_values; ++t) {
                result += values[t];
            }
            result /= float(num_values);
            break;
    }

    // Save the result to the correct pixel location.
    results[trj_offset + pixel_index] = result;
}

void deviceGetCoadds(const unsigned int num_images, const unsigned int width, const unsigned int height,
                     std::vector<float *> data_refs, PerImageData image_data, int num_trajectories,
                     Trajectory *trajectories, StampParameters params,
                     std::vector<std::vector<bool>> &use_index_vect, float *results) {
    // Allocate Device memory
    Trajectory *device_trjs;
    int *device_use_index = nullptr;
    float *device_times;
    float *device_img;
    float *device_res;

    // Compute the dimensions for the data.
    const unsigned int num_image_pixels = num_images * width * height;
    const unsigned int stamp_width = 2 * params.radius + 1;
    const unsigned int stamp_ppi = (2 * params.radius + 1) * (2 * params.radius + 1);
    const unsigned int num_stamp_pixels = num_trajectories * stamp_ppi;

    // Allocate and copy the trajectories.
    checkCudaErrors(cudaMalloc((void **)&device_trjs, sizeof(Trajectory) * num_trajectories));
    checkCudaErrors(cudaMemcpy(device_trjs, trajectories, sizeof(Trajectory) * num_trajectories,
                               cudaMemcpyHostToDevice));

    // Check if we need to create a vector of per-trajectory, per-image use.
    // Convert the vector of booleans into an integer array so we do a cudaMemcpy.
    if (use_index_vect.size() == num_trajectories) {
        checkCudaErrors(cudaMalloc((void **)&device_use_index, sizeof(int) * num_images * num_trajectories));

        int *start_ptr = device_use_index;
        std::vector<int> int_vect(num_images, 0);
        for (unsigned i = 0; i < num_trajectories; ++i) {
            assertm(use_index_vect[i].size() == num_images,
                    "Number of images and indices selected for processing do not match");
            for (unsigned t = 0; t < num_images; ++t) {
                int_vect[t] = use_index_vect[i][t] ? 1 : 0;
            }

            checkCudaErrors(
                    cudaMemcpy(start_ptr, int_vect.data(), sizeof(int) * num_images, cudaMemcpyHostToDevice));
            start_ptr += num_images;
        }
    }

    // Allocate and copy the times.
    checkCudaErrors(cudaMalloc((void **)&device_times, sizeof(float) * num_images));
    checkCudaErrors(cudaMemcpy(device_times, image_data.image_times, sizeof(float) * num_images,
                               cudaMemcpyHostToDevice));

    // Allocate and copy the images.
    checkCudaErrors(cudaMalloc((void **)&device_img, sizeof(float) * num_image_pixels));
    float *next_ptr = device_img;
    for (unsigned t = 0; t < num_images; ++t) {
        checkCudaErrors(
                cudaMemcpy(next_ptr, data_refs[t], sizeof(float) * width * height, cudaMemcpyHostToDevice));
        next_ptr += width * height;
    }

    // Allocate space for the results.
    checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(float) * num_stamp_pixels));

    // Wrap the per-image data into a struct. This struct will be copied by value
    // during the function call, so we don't need to allocate memory for the
    // struct itself. We just set the pointers to the on device vectors.
    PerImageData device_image_data;
    device_image_data.num_images = num_images;
    device_image_data.image_times = device_times;
    device_image_data.psi_params = nullptr;
    device_image_data.phi_params = nullptr;

    dim3 blocks(num_trajectories, 1, 1);
    dim3 threads(1, stamp_width, stamp_width);

    // Create the stamps.
    deviceGetCoaddStamp<<<blocks, threads>>>(num_images, width, height, device_img, device_image_data,
                                             num_trajectories, device_trjs, params, device_use_index,
                                             device_res);
    cudaDeviceSynchronize();

    // Free up the unneeded memory (everything except for the on-device results).
    checkCudaErrors(cudaFree(device_img));
    if (device_use_index != nullptr) checkCudaErrors(cudaFree(device_use_index));
    checkCudaErrors(cudaFree(device_times));
    checkCudaErrors(cudaFree(device_trjs));
    cudaDeviceSynchronize();

    // Read back results
    checkCudaErrors(
            cudaMemcpy(results, device_res, sizeof(float) * num_stamp_pixels, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // Free the rest of the  on GPU memory.
    checkCudaErrors(cudaFree(device_res));
}

/*
    void deviceGetCoadds(ImageStack &stack, PerImageData image_data, int num_trajectories,
    Trajectory *trajectories, StampParameters params,
    std::vector<std::vector<bool>> &use_index_vect, float *results) {
    // Allocate Device memory
    Trajectory *device_trjs;
    int *device_use_index = nullptr;
    float *device_times;
    float *device_img;
    float *device_res;

    // Compute the dimensions for the data.
    const unsigned int num_images = stack.img_count();
    const unsigned int width = stack.get_width();
    const unsigned int height = stack.get_height();
    const unsigned int num_image_pixels = num_images * width * height;
    const unsigned int stamp_width = 2 * params.radius + 1;
    const unsigned int stamp_ppi = (2 * params.radius + 1) * (2 * params.radius + 1);
    const unsigned int num_stamp_pixels = num_trajectories * stamp_ppi;

    // Allocate and copy the trajectories.
    checkCudaErrors(cudaMalloc((void **)&device_trjs, sizeof(Trajectory) * num_trajectories));
    checkCudaErrors(cudaMemcpy(device_trjs, trajectories, sizeof(Trajectory) * num_trajectories,
    cudaMemcpyHostToDevice));

    // Check if we need to create a vector of per-trajectory, per-image use.
    // Convert the vector of booleans into an integer array so we do a cudaMemcpy.
    if (use_index_vect.size() == num_trajectories) {
    checkCudaErrors(cudaMalloc((void **)&device_use_index, sizeof(int) * num_images * num_trajectories));

    int *start_ptr = device_use_index;
    std::vector<int> int_vect(num_images, 0);
    for (unsigned i = 0; i < num_trajectories; ++i) {
    assert(use_index_vect[i].size() == num_images);
    for (unsigned t = 0; t < num_images; ++t) {
    int_vect[t] = use_index_vect[i][t] ? 1 : 0;
    }

    checkCudaErrors(
    cudaMemcpy(start_ptr, int_vect.data(), sizeof(int) * num_images, cudaMemcpyHostToDevice));
    start_ptr += num_images;
    }
    }

    // Allocate and copy the times.
    checkCudaErrors(cudaMalloc((void **)&device_times, sizeof(float) * num_images));
    checkCudaErrors(cudaMemcpy(device_times, image_data.image_times, sizeof(float) * num_images,
    cudaMemcpyHostToDevice));

    // Allocate and copy the images.
    checkCudaErrors(cudaMalloc((void **)&device_img, sizeof(float) * num_image_pixels));
    float *next_ptr = device_img;
    for (unsigned t = 0; t < num_images; ++t) {
    // Used to be a vector of floats, now is an eigen::vector of floats or something
    // but that's ok because all we use it for is the .data() -> float*
    // I think this can also just directly go to .data because of RowMajor layout
    auto& data_ref = stack.get_single_image(t).get_science().get_image();

    assert(data_ref.size() == width * height);
    checkCudaErrors(cudaMemcpy(next_ptr, data_ref.data(), sizeof(float) * width * height,
    cudaMemcpyHostToDevice));
    next_ptr += width * height;
    }

    // Allocate space for the results.
    checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(float) * num_stamp_pixels));

    // Wrap the per-image data into a struct. This struct will be copied by value
    // during the function call, so we don't need to allocate memory for the
    // struct itself. We just set the pointers to the on device vectors.
    PerImageData device_image_data;
    device_image_data.num_images = num_images;
    device_image_data.image_times = device_times;
    device_image_data.psi_params = nullptr;
    device_image_data.phi_params = nullptr;

    dim3 blocks(num_trajectories, 1, 1);
    dim3 threads(1, stamp_width, stamp_width);

    // Create the stamps.
    deviceGetCoaddStamp<<<blocks, threads>>>(num_images, width, height, device_img, device_image_data,
    num_trajectories, device_trjs, params, device_use_index,
    device_res);
    cudaDeviceSynchronize();

    // Free up the unneeded memory (everything except for the on-device results).
    checkCudaErrors(cudaFree(device_img));
    if (device_use_index != nullptr) checkCudaErrors(cudaFree(device_use_index));
    checkCudaErrors(cudaFree(device_times));
    checkCudaErrors(cudaFree(device_trjs));
    cudaDeviceSynchronize();

    // Read back results
    checkCudaErrors(
    cudaMemcpy(results, device_res, sizeof(float) * num_stamp_pixels, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // Free the rest of the  on GPU memory.
    checkCudaErrors(cudaFree(device_res));
    }
*/
} /* namespace search */

#endif /* KERNELS_CU_ */
