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

//#include "image_stack.h"

namespace search {

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

  __device__ float ReadEncodedPixel(void *image_vect, int index, int n_bytes, const scaleParameters &params) {
    float value = (n_bytes == 1) ? (float)reinterpret_cast<uint8_t *>(image_vect)[index]
      : (float)reinterpret_cast<uint16_t *>(image_vect)[index];
    float result = (value == 0.0) ? NO_DATA : (value - 1.0) * params.scale + params.min_val;
    return result;
  }

  /*
   * Searches through images (represented as a flat array of floats) looking for most likely
   * trajectories in the given list. Outputs a results image of best trajectories. Returns a
   * fixed number of results per pixel specified by RESULTS_PER_PIXEL
   * filters results using a sigma_g-based filter and a central-moment filter.
   */
  __global__ void searchFilterImages(int num_images, int width, int height, void *psi_vect, void *phi_vect,
                                     PerImageData image_data, SearchParameters params, int num_trajectories,
                                     Trajectory *trajectories, Trajectory *results) {
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
    // TODO: this is an ugly hack to get things to work,
    // beautify before merge, see also later
    const int y = x_i + params.x_start_min;
    const int x = y_i + params.y_start_min;
    const unsigned int n_pixels = width * height;

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
      for (int i = 0; i < num_images; ++i) {
        lc_array[i] = 0;
        psi_array[i] = 0;
        phi_array[i] = 0;
        idx_array[i] = i;
      }

      // Loop over each image and sample the appropriate pixel
      int num_seen = 0;
      for (int i = 0; i < num_images; ++i) {
        // Predict the trajectory's position.
        float curr_time = image_data.image_times[i];
        // TODO: the hack again, make sure to properly contextualize
        // before merging
        int current_y = x + int(curr_trj.vx * curr_time + 0.5);
        int current_x = y + int(curr_trj.vy * curr_time + 0.5);

        // Test if trajectory goes out of the image, in which case we do not
        // look up a pixel value for this time step (allowing trajectories to
        // overlap the image for only some of the times).
        if (current_x >= width || current_y >= height || current_x < 0 || current_y < 0) {
          continue;
        }

        // Get the Psi and Phi pixel values.
        unsigned int pixel_index = (n_pixels * i + current_x * width + current_y);
        float curr_psi = (params.psi_num_bytes <= 0 || image_data.psi_params == nullptr)
          ? reinterpret_cast<float *>(psi_vect)[pixel_index]
          : ReadEncodedPixel(psi_vect, pixel_index, params.psi_num_bytes,
                             image_data.psi_params[i]);
        if (curr_psi == NO_DATA) continue;

        float curr_phi = (params.phi_num_bytes <= 0 || image_data.phi_params == nullptr)
          ? reinterpret_cast<float *>(phi_vect)[pixel_index]
          : ReadEncodedPixel(phi_vect, pixel_index, params.phi_num_bytes,
                             image_data.phi_params[i]);
        if (curr_phi == NO_DATA) continue;

        if (curr_psi != NO_DATA && curr_phi != NO_DATA) {
          curr_trj.obs_count++;
          psi_sum += curr_psi;
          phi_sum += curr_phi;
          psi_array[num_seen] = curr_psi;
          phi_array[num_seen] = curr_phi;
          if (curr_phi != 0.0) lc_array[num_seen] = curr_psi / curr_phi;
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

  template <typename T>
  void *encodeImage(float *image_vect, int num_times, int num_pixels, scaleParameters *params, bool debug) {
    void *device_vect = NULL;

    long unsigned int total_size = sizeof(T) * num_times * num_pixels;
    if (debug) {
      printf("Encoding image into %lu bytes/pixel for a total of %lu bytes.\n", sizeof(T), total_size);
    }

    // Do the encoding locally first.
    T *encoded = (T *)malloc(total_size);
    for (int t = 0; t < num_times; ++t) {
      float safe_max = params[t].max_val - params[t].scale / 100.0;
      for (int p = 0; p < num_pixels; ++p) {
        int index = t * num_pixels + p;
        float value = image_vect[index];
        if (value == NO_DATA) {
          encoded[index] = 0;
        } else {
          value = min(value, safe_max);
          value = max(value, params[t].min_val);
          value = (value - params[t].min_val) / params[t].scale + 1.0;
          encoded[index] = static_cast<T>(value);
        }
      }
    }

    // Allocate the space on device and do a direct copy.
    checkCudaErrors(cudaMalloc((void **)&device_vect, total_size));
    checkCudaErrors(cudaMemcpy(device_vect, encoded, total_size, cudaMemcpyHostToDevice));

    // Free the local space.
    free(encoded);

    return device_vect;
  }

  void *encodeImageFloat(float *image_vect, unsigned int vectLength, bool debug) {
    void *device_vect = NULL;
    long unsigned int total_size = sizeof(float) * vectLength;

    if (debug) {
      printf("Encoding image as float for a total of %lu bytes.\n", total_size);
    }

    checkCudaErrors(cudaMalloc((void **)&device_vect, total_size));
    checkCudaErrors(cudaMemcpy(device_vect, image_vect, total_size, cudaMemcpyHostToDevice));
    return device_vect;
  }

  extern "C" void deviceSearchFilter(int num_images, int width, int height, float *psi_vect, float *phi_vect,
                                     PerImageData img_data, SearchParameters params, int num_trajectories,
                                     Trajectory *trj_to_search, int num_results, Trajectory *best_results) {
    // Allocate Device memory
    Trajectory *device_tests;
    float *device_img_times;
    void *device_psi;
    void *device_phi;
    Trajectory *device_search_results;
    scaleParameters *device_psi_params = nullptr;
    scaleParameters *device_phi_params = nullptr;

    // Check the hard coded maximum number of images against the num_images.
    if (num_images > MAX_NUM_IMAGES) {
      throw std::runtime_error("Number of images exceeds GPU maximum.");
    }

    if (params.debug) {
      printf("Allocating %lu bytes for testing grid.\n", sizeof(Trajectory) * num_trajectories);
    }
    checkCudaErrors(cudaMalloc((void **)&device_tests, sizeof(Trajectory) * num_trajectories));

    if (params.debug) {
      printf("Allocating %lu bytes for time data.\n", sizeof(float) * num_images);
    }
    checkCudaErrors(cudaMalloc((void **)&device_img_times, sizeof(float) * num_images));

    if (params.debug) {
      printf("Allocating %lu bytes for testing grid.\n", sizeof(Trajectory) * num_trajectories);
    }
    checkCudaErrors(cudaMalloc((void **)&device_search_results, sizeof(Trajectory) * num_results));

    // Copy trajectories to search
    checkCudaErrors(cudaMemcpy(device_tests, trj_to_search, sizeof(Trajectory) * num_trajectories,
                               cudaMemcpyHostToDevice));

    // Copy image times
    checkCudaErrors(cudaMemcpy(device_img_times, img_data.image_times, sizeof(float) * num_images,
                               cudaMemcpyHostToDevice));

    // Copy (and encode) the images. Also copy over the scaling parameters if needed.
    if ((params.psi_num_bytes == 1 || params.psi_num_bytes == 2) && (img_data.psi_params != nullptr)) {
      checkCudaErrors(cudaMalloc((void **)&device_psi_params, num_images * sizeof(scaleParameters)));
      checkCudaErrors(cudaMemcpy(device_psi_params, img_data.psi_params,
                                 num_images * sizeof(scaleParameters), cudaMemcpyHostToDevice));
      if (params.psi_num_bytes == 1) {
        device_psi = encodeImage<uint8_t>(psi_vect, num_images, width * height, img_data.psi_params,
                                          params.debug);
      } else {
        device_psi = encodeImage<uint16_t>(psi_vect, num_images, width * height, img_data.psi_params,
                                           params.debug);
      }
    } else {
      device_psi = encodeImageFloat(psi_vect, num_images * width * height, params.debug);
    }
    if ((params.phi_num_bytes == 1 || params.phi_num_bytes == 2) && (img_data.phi_params != nullptr)) {
      checkCudaErrors(cudaMalloc((void **)&device_phi_params, num_images * sizeof(scaleParameters)));
      checkCudaErrors(cudaMemcpy(device_phi_params, img_data.phi_params,
                                 num_images * sizeof(scaleParameters), cudaMemcpyHostToDevice));
      if (params.phi_num_bytes == 1) {
        device_phi = encodeImage<uint8_t>(phi_vect, num_images, width * height, img_data.phi_params,
                                          params.debug);
      } else {
        device_phi = encodeImage<uint16_t>(phi_vect, num_images, width * height, img_data.phi_params,
                                           params.debug);
      }
    } else {
      device_phi = encodeImageFloat(phi_vect, num_images * width * height, params.debug);
    }

    // Wrap the per-image data into a struct. This struct will be copied by value
    // during the function call, so we don't need to allocate memory for the
    // struct itself. We just set the pointers to the on device vectors.
    PerImageData device_image_data;
    device_image_data.num_images = num_images;
    device_image_data.image_times = device_img_times;
    device_image_data.psi_params = device_psi_params;
    device_image_data.phi_params = device_phi_params;

    // Compute the range of starting pixels to use when setting the blocks and threads.
    // We use the width and height of the search space (as opposed to the image width
    // and height), meaning the blocks/threads will be indexed relative to the search space.
    int search_width = params.x_start_max - params.x_start_min;
    int search_height = params.y_start_max - params.y_start_min;
    dim3 blocks(search_width / THREAD_DIM_X + 1, search_height / THREAD_DIM_Y + 1);
    dim3 threads(THREAD_DIM_X, THREAD_DIM_Y);

    // Launch Search
    searchFilterImages<<<blocks, threads>>>(num_images, width, height, device_psi, device_phi,
                                            device_image_data, params, num_trajectories, device_tests,
                                            device_search_results);

    // Read back results
    checkCudaErrors(cudaMemcpy(best_results, device_search_results, sizeof(Trajectory) * num_results,
                               cudaMemcpyDeviceToHost));

    // Free the on GPU memory.
    if (device_phi_params != nullptr) checkCudaErrors(cudaFree(device_phi_params));
    if (device_psi_params != nullptr) checkCudaErrors(cudaFree(device_psi_params));
    checkCudaErrors(cudaFree(device_phi));
    checkCudaErrors(cudaFree(device_psi));
    checkCudaErrors(cudaFree(device_search_results));
    checkCudaErrors(cudaFree(device_img_times));
    checkCudaErrors(cudaFree(device_tests));
  }

  __global__ void deviceGetCoaddStamp(int num_images, int width, int height, float *image_vect,
                                      PerImageData image_data, int num_trajectories, Trajectory *trajectories,
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
      float curr_time = image_data.image_times[t];
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

    
  void deviceGetCoadds(const unsigned int num_images,
		       const unsigned int height,
		       const unsigned int width,
		       std::vector<float*> data_refs,
		       PerImageData image_data,
		       int num_trajectories,
                       Trajectory *trajectories,
		       StampParameters params,
                       std::vector<std::vector<bool>> &use_index_vect,
		       float *results) {
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
        assertm(use_index_vect[i].size() == num_images, "Number of images and indices selected for processing do not match");
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
      checkCudaErrors(cudaMemcpy(next_ptr, data_refs[t], sizeof(float) * width * height,
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
