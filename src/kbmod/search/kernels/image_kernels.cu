/*
 * image_kernels.cu
 *
 * Created on: October 10, 2022
 * (Split from kernels.cu)
 */

#ifndef IMAGE_KERNELS_CU_
#define IMAGE_KERNELS_CU_

#include "../common.h"
#include "cuda_errors.h"
#include <stdio.h>
#include <float.h>
#include <stdexcept>

namespace search {

// This is defined in kernels.cu.
__host__ __device__ bool device_pixel_valid(float value);

/*
 * Device kernel that convolves the provided image with the psf
 */
__global__ void convolve_psf(int width, int height, float *source_img, float *result_img,
                             float *psf, int psf_radius, int psf_dim, float psf_sum) {
    // Find bounds of convolution area
    const int x = blockIdx.x * CONV_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * CONV_THREAD_DIM + threadIdx.y;
    if (x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
    const uint64_t result_index = y * width + x;

    // Read kernel
    float sum = 0.0;
    float psf_portion = 0.0;
    float center = source_img[y * width + x];
    if (device_pixel_valid(center)) {
        for (int j = -psf_radius; j <= psf_radius; j++) {
            // #pragma unroll
            for (int i = -psf_radius; i <= psf_radius; i++) {
                if ((x + i >= 0) && (x + i < width) && (y + j >= 0) && (y + j < height)) {
                    float current_pix = source_img[(y + j) * width + (x + i)];
                    if (device_pixel_valid(current_pix)) {
                        float current_psf = psf[(j + psf_radius) * psf_dim + (i + psf_radius)];
                        psf_portion += current_psf;
                        sum += current_pix * current_psf;
                    }
                }
            }
        }

        result_img[result_index] = (psf_portion != 0.0) ? (sum * psf_sum) / psf_portion : 0.0;
    } else {
        // Leave masked and NaN pixels alone (these could be replaced here with zero)
        result_img[result_index] = center;  // 0.0
    }
}

extern "C" void deviceConvolve(float *source_img, float *result_img, int width, int height, float *psf_kernel,
                               int psf_radius, float psf_sum) {
    if (width <= 0) throw std::runtime_error("Invalid width = " + std::to_string(width));
    if (height <= 0) throw std::runtime_error("Invalid height = " + std::to_string(width));
    if (psf_radius < 0) throw std::runtime_error("Invalid PSF radius = " + std::to_string(psf_radius));

    uint64_t n_pixels = width * height;
    int psf_dim = 2 * psf_radius + 1;
    int psf_size = psf_dim * psf_dim;

    // Pointers to device memory
    float *device_kernel;
    float *devicesource_img;
    float *deviceresult_img;

    dim3 blocks(width / CONV_THREAD_DIM + 1, height / CONV_THREAD_DIM + 1);
    dim3 threads(CONV_THREAD_DIM, CONV_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&device_kernel, sizeof(float) * psf_size));
    checkCudaErrors(cudaMalloc((void **)&devicesource_img, sizeof(float) * n_pixels));
    checkCudaErrors(cudaMalloc((void **)&deviceresult_img, sizeof(float) * n_pixels));

    checkCudaErrors(cudaMemcpy(device_kernel, psf_kernel, sizeof(float) * psf_size, cudaMemcpyHostToDevice));

    checkCudaErrors(
            cudaMemcpy(devicesource_img, source_img, sizeof(float) * n_pixels, cudaMemcpyHostToDevice));

    convolve_psf<<<blocks, threads>>>(width, height, devicesource_img, deviceresult_img, device_kernel,
                                      psf_radius, psf_dim, psf_sum);

    checkCudaErrors(
            cudaMemcpy(result_img, deviceresult_img, sizeof(float) * n_pixels, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(device_kernel));
    checkCudaErrors(cudaFree(devicesource_img));
    checkCudaErrors(cudaFree(deviceresult_img));
}

} /* namespace search */

#endif /* IMAGE_KERNELS_CU_ */
