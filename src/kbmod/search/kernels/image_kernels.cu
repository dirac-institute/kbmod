/*
 * image_kernels.cu
 *
 * Created on: October 10, 2022
 * (Split from kernels.cu)
 */

#ifndef IMAGE_KERNELS_CU_
#define IMAGE_KERNELS_CU_

#include <stdio.h>
#include <float.h>
#include <stdexcept>
#include <vector>

#include "../common.h"
#include "../gpu_array.h"

namespace search {

// This is defined in kernels.cu.
__host__ __device__ bool device_pixel_valid(float value);

/*
 * Device kernel that compute the autocorrelation of the PSF and the image.
 * Although we call this convolve (by astronomy conventions) autocorrelation is a
 * flipped version of standard convolution.
 */
__global__ void convolve_psf(int width, int height, float *source_img, float *result_img, float *psf,
                             int psf_radius, int psf_dim, float psf_sum) {
    // Find bounds of convolution area
    const int x = blockIdx.x * CONV_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * CONV_THREAD_DIM + threadIdx.y;
    if (x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
    const uint64_t result_index = y * width + x;
    const uint64_t total_img_pixels = height * width;
    if (result_index >= total_img_pixels) {
        // This is an error condition that should never happen.
        return;
    }

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

extern "C" void deviceConvolve(float *source_img, float *result_img, int width, int height,
                               float *psf_kernel, int psf_radius) {
    if (width <= 0) throw std::runtime_error("Invalid width = " + std::to_string(width));
    if (height <= 0) throw std::runtime_error("Invalid height = " + std::to_string(height));
    if (psf_radius < 0) throw std::runtime_error("Invalid PSF radius = " + std::to_string(psf_radius));

    uint64_t n_pixels = width * height;
    int psf_dim = 2 * psf_radius + 1;
    
    // Compute the PSF sum.
    float psf_sum = 0.0;
    for (int r = 0; r < psf_dim; ++r) {
        for (int c = 0; c < psf_dim; ++c) 
            psf_sum += psf_kernel[r * psf_dim + c]
        }
    }

    // Allocate Device memory
    float *device_kernel;
    checkCudaErrors(cudaMalloc((void **)&device_kernel, sizeof(float) * psf_size));
    GPUArray<float> devicesource_img(n_pixels, true);
    GPUArray<float> deviceresult_img(n_pixels, true);

    // Copy the source image and the PSF.
    devicesource_img.copy_array_into_subset_of_gpu(source_img, 0, n_pixels);
    checkCudaErrors(cudaMemcpy(device_kernel, psf_kernel, sizeof(float) * psf_size, cudaMemcpyHostToDevice));

    dim3 blocks(width / CONV_THREAD_DIM + 1, height / CONV_THREAD_DIM + 1);
    dim3 threads(CONV_THREAD_DIM, CONV_THREAD_DIM);
    convolve_psf<<<blocks, threads>>>(width, height, devicesource_img.get_ptr(), deviceresult_img.get_ptr(),
                                      device_kernel.get_ptr(), psf_radius, psf_dim, psf_sum);

    // Copy the result image off the GPU.
    deviceresult_img.copy_subset_of_gpu_into_array(result_img, 0, n_pixels);

    // Free all the on-device memory.
    checkCudaErrors(cudaFree(device_kernel));
    devicesource_img.free_gpu_memory();
    deviceresult_img.free_gpu_memory();
}

} /* namespace search */

#endif /* IMAGE_KERNELS_CU_ */
