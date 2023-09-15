/*
 * image_kernels.cu
 *
 * Created on: October 10, 2022
 * (Split from kernels.cu)
 */

#ifndef IMAGE_KERNELS_CU_
#define IMAGE_KERNELS_CU_

#define MAX_STAMP_IMAGES 200

#include <assert.h>
#include "common.h"
#include "cuda_errors.h"
#include <stdio.h>
#include <float.h>

#include "ImageStack.h"

namespace search {

/*
 * Device kernel that convolves the provided image with the psf
 */
__global__ void convolvePSF(int width, int height, float *sourceImage, float *resultImage, float *psf,
                            int psfRad, int psfDim, float psfSum, float maskFlag) {
    // Find bounds of convolution area
    const int x = blockIdx.x * CONV_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * CONV_THREAD_DIM + threadIdx.y;
    if (x < 0 || x > width - 1 || y < 0 || y > height - 1) return;

    // Read kernel
    float sum = 0.0;
    float psfPortion = 0.0;
    float center = sourceImage[y * width + x];
    if (center != NO_DATA) {
        for (int j = -psfRad; j <= psfRad; j++) {
            // #pragma unroll
            for (int i = -psfRad; i <= psfRad; i++) {
                if ((x + i >= 0) && (x + i < width) && (y + j >= 0) && (y + j < height)) {
                    float currentPixel = sourceImage[(y + j) * width + (x + i)];
                    if (currentPixel != NO_DATA) {
                        float currentPSF = psf[(j + psfRad) * psfDim + (i + psfRad)];
                        psfPortion += currentPSF;
                        sum += currentPixel * currentPSF;
                    }
                }
            }
        }

        resultImage[y * width + x] = (sum * psfSum) / psfPortion;
    } else {
        // Leave masked pixel alone (these could be replaced here with zero)
        resultImage[y * width + x] = NO_DATA;  // 0.0
    }
}

extern "C" void deviceConvolve(float *sourceImg, float *resultImg, int width, int height, float *psfKernel,
                               int psfSize, int psfDim, int psfRadius, float psfSum) {
    // Pointers to device memory
    float *deviceKernel;
    float *deviceSourceImg;
    float *deviceResultImg;

    long pixelsPerImage = width * height;
    dim3 blocks(width / CONV_THREAD_DIM + 1, height / CONV_THREAD_DIM + 1);
    dim3 threads(CONV_THREAD_DIM, CONV_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceKernel, sizeof(float) * psfSize));
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg, sizeof(float) * pixelsPerImage));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg, sizeof(float) * pixelsPerImage));

    checkCudaErrors(cudaMemcpy(deviceKernel, psfKernel, sizeof(float) * psfSize, cudaMemcpyHostToDevice));

    checkCudaErrors(
            cudaMemcpy(deviceSourceImg, sourceImg, sizeof(float) * pixelsPerImage, cudaMemcpyHostToDevice));

    convolvePSF<<<blocks, threads>>>(width, height, deviceSourceImg, deviceResultImg, deviceKernel, psfRadius,
                                     psfDim, psfSum, NO_DATA);

    checkCudaErrors(
            cudaMemcpy(resultImg, deviceResultImg, sizeof(float) * pixelsPerImage, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceKernel));
    checkCudaErrors(cudaFree(deviceSourceImg));
    checkCudaErrors(cudaFree(deviceResultImg));
}

__global__ void device_get_coadd_stamp(int num_images, int width, int height, float *image_vect,
                                       perImageData image_data, int num_trajectories,
                                       trajectory *trajectories, stampParameters params, int *use_index_vect,
                                       float *results) {
    // Get the trajectory that we are going to be using.
    const int trj_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (trj_index < 0 || trj_index >= num_trajectories) return;
    trajectory trj = trajectories[trj_index];

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
    assert(num_images < MAX_STAMP_IMAGES);
    float values[MAX_STAMP_IMAGES];

    // Loop over each image and compute the stamp.
    int num_values = 0;
    for (int t = 0; t < num_images; ++t) {
        // Skip entries marked 0 in the use_index_vect.
        if (use_index_vect != nullptr && use_index_vect[use_index_offset + t] == 0) {
            continue;
        }

        // Predict the trajectory's position including the barycentric correction if needed.
        float cTime = image_data.imageTimes[t];
        int currentX = int(trj.x + trj.xVel * cTime);
        int currentY = int(trj.y + trj.yVel * cTime);
        if (image_data.baryCorrs != nullptr) {
            baryCorrection bc = image_data.baryCorrs[t];
            currentX = int(trj.x + trj.xVel * cTime + bc.dx + trj.x * bc.dxdx + trj.y * bc.dxdy);
            currentY = int(trj.y + trj.yVel * cTime + bc.dy + trj.x * bc.dydx + trj.y * bc.dydy);
        }

        // Get the stamp and add it to the list of values.
        int img_x = currentX - params.radius + stamp_x;
        int img_y = currentY - params.radius + stamp_y;
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

void deviceGetCoadds(ImageStack &stack, perImageData image_data, int num_trajectories,
                     trajectory *trajectories, stampParameters params,
                     std::vector<std::vector<bool>> &use_index_vect, float *results) {
    // Allocate Device memory
    trajectory *device_trjs;
    int *device_use_index = nullptr;
    float *device_times;
    float *device_img;
    float *device_res;
    baryCorrection *deviceBaryCorrs = nullptr;

    // Compute the dimensions for the data.
    const unsigned int num_images = stack.imgCount();
    const unsigned int width = stack.getWidth();
    const unsigned int height = stack.getHeight();
    const unsigned int num_image_pixels = num_images * width * height;
    const unsigned int stamp_width = 2 * params.radius + 1;
    const unsigned int stamp_ppi = (2 * params.radius + 1) * (2 * params.radius + 1);
    const unsigned int num_stamp_pixels = num_trajectories * stamp_ppi;

    // Allocate and copy the trajectories.
    checkCudaErrors(cudaMalloc((void **)&device_trjs, sizeof(trajectory) * num_trajectories));
    checkCudaErrors(cudaMemcpy(device_trjs, trajectories, sizeof(trajectory) * num_trajectories,
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
    checkCudaErrors(cudaMemcpy(device_times, image_data.imageTimes, sizeof(float) * num_images,
                               cudaMemcpyHostToDevice));

    // Allocate and copy the images.
    checkCudaErrors(cudaMalloc((void **)&device_img, sizeof(float) * num_image_pixels));
    float *next_ptr = device_img;
    for (unsigned t = 0; t < num_images; ++t) {
        const std::vector<float> &data_ref = stack.getSingleImage(t).getScience().getPixels();

        assert(data_ref.size() == width * height);
        checkCudaErrors(cudaMemcpy(next_ptr, data_ref.data(), sizeof(float) * width * height,
                                   cudaMemcpyHostToDevice));
        next_ptr += width * height;
    }

    // Allocate space for the results.
    checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(float) * num_stamp_pixels));

    // Allocate memory for and copy barycentric corrections (if needed).
    if (image_data.baryCorrs != nullptr) {
        checkCudaErrors(cudaMalloc((void **)&deviceBaryCorrs, sizeof(baryCorrection) * num_images));
        checkCudaErrors(cudaMemcpy(deviceBaryCorrs, image_data.baryCorrs, sizeof(baryCorrection) * num_images,
                                   cudaMemcpyHostToDevice));
    }

    // Wrap the per-image data into a struct. This struct will be copied by value
    // during the function call, so we don't need to allocate memory for the
    // struct itself. We just set the pointers to the on device vectors.
    perImageData device_image_data;
    device_image_data.numImages = num_images;
    device_image_data.imageTimes = device_times;
    device_image_data.baryCorrs = deviceBaryCorrs;
    device_image_data.psiParams = nullptr;
    device_image_data.phiParams = nullptr;

    dim3 blocks(num_trajectories, 1, 1);
    dim3 threads(1, stamp_width, stamp_width);

    // Create the stamps.
    device_get_coadd_stamp<<<blocks, threads>>>(num_images, width, height, device_img, device_image_data,
                                                num_trajectories, device_trjs, params, device_use_index,
                                                device_res);
    cudaDeviceSynchronize();

    // Free up the unneeded memory (everything except for the on-device results).
    if (deviceBaryCorrs != nullptr) checkCudaErrors(cudaFree(deviceBaryCorrs));
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

} /* namespace search */

#endif /* IMAGE_KERNELS_CU_ */
