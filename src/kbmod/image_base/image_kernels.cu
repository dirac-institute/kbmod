/*
 * image_kernels.cu
 *
 * Created on: October 10, 2022
 * (Split from kernels.cu)
 */

#ifndef IMAGE_KERNELS_CU_
#define IMAGE_KERNELS_CU_

constexpr unsigned short CONV_THREAD_DIM = 32;

#include <assert.h>
#include "common.h"
#include "cuda_errors.h"
#include <stdio.h>
#include <float.h>

namespace image_base {

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

} /* namespace image_base */

#endif /* IMAGE_KERNELS_CU_ */
