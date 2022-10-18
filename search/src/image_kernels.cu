/*
 * image_kernels.cu
 *
 * Created on: October 10, 2022
 * (Split from kernels.cu)
 */

#ifndef IMAGE_KERNELS_CU_
#define IMAGE_KERNELS_CU_

#include "common.h"
#include <helper_cuda.h>
#include <stdio.h>
#include <float.h>

namespace kbmod {

/*
 * Device kernel that convolves the provided image with the psf
 */
__global__ void convolvePSF(int width, int height, float *sourceImage,
                            float *resultImage, float *psf, int psfRad, int psfDim,
                            float psfSum, float maskFlag) {
    // Find bounds of convolution area
    const int x = blockIdx.x*CONV_THREAD_DIM+threadIdx.x;
    const int y = blockIdx.y*CONV_THREAD_DIM+threadIdx.y;
    if (x < 0 || x > width-1 || y < 0 || y > height-1) return;

    // Read kernel
    float sum = 0.0;
    float psfPortion = 0.0;
    float center = sourceImage[y*width+x];
    if (center != NO_DATA) {
        for (int j = -psfRad; j <= psfRad; j++) {
            // #pragma unroll
            for (int i = -psfRad; i <= psfRad; i++) {
                if ((x + i >= 0) && (x + i < width) &&
                    (y + j >= 0) && (y + j < height)) {
                    float currentPixel = sourceImage[(y + j) * width + (x + i)];
                    if (currentPixel != NO_DATA) {
                        float currentPSF = psf[(j + psfRad) * psfDim + ( i + psfRad)];
                        psfPortion += currentPSF;
                        sum += currentPixel * currentPSF;
                    }
                }
            }
        }

        resultImage[y*width+x] = (sum * psfSum) / psfPortion;
    } else {
        // Leave masked pixel alone (these could be replaced here with zero)
        resultImage[y*width+x] = NO_DATA; // 0.0
    }
}

extern "C" void deviceConvolve(float *sourceImg, float *resultImg,
                               int width, int height, float *psfKernel,
                               int psfSize, int psfDim, int psfRadius,
                               float psfSum) {
    // Pointers to device memory
    float *deviceKernel;
    float *deviceSourceImg;
    float *deviceResultImg;

    long pixelsPerImage = width*height;
    dim3 blocks(width / CONV_THREAD_DIM + 1, height / CONV_THREAD_DIM + 1);
    dim3 threads(CONV_THREAD_DIM, CONV_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceKernel, sizeof(float) * psfSize));
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg, sizeof(float) * pixelsPerImage));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg, sizeof(float) * pixelsPerImage));

    checkCudaErrors(cudaMemcpy(deviceKernel, psfKernel,
        sizeof(float)*psfSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(deviceSourceImg, sourceImg,
        sizeof(float)*pixelsPerImage, cudaMemcpyHostToDevice));

    convolvePSF<<<blocks, threads>>> (width, height, deviceSourceImg,
        deviceResultImg, deviceKernel, psfRadius, psfDim, psfSum, NO_DATA);

    checkCudaErrors(cudaMemcpy(resultImg, deviceResultImg,
        sizeof(float)*pixelsPerImage, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceKernel));
    checkCudaErrors(cudaFree(deviceSourceImg));
    checkCudaErrors(cudaFree(deviceResultImg));
}

// Reads a single pixel from an image buffer
__device__ float readPixel(float* img, int x, int y, int width, int height) {
    return (x < width && y < height) ? img[y * width + x] : NO_DATA;
}

__device__ float maxMasked(float pixel, float previousMax) {
    return pixel == NO_DATA ? previousMax : max(pixel, previousMax);
}

__device__ float minMasked(float pixel, float previousMin) {
    return pixel == NO_DATA ? previousMin : min(pixel, previousMin);
}

/*
 * Reduces the resolution of an image to 1/4 using max pooling
 */
__global__ void pool(int sourceWidth, int sourceHeight, float *source,
                     int destWidth, int destHeight, float *dest, short mode)
{
    const int x = blockIdx.x * POOL_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * POOL_THREAD_DIM + threadIdx.y;
    if (x >= destWidth || y >= destHeight)
        return;

    float mp;
    float pixel;
    if (mode == POOL_MAX) {
        mp = -FLT_MAX;
        pixel = readPixel(source, 2 * x, 2 * y, sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        pixel = readPixel(source, 2 * x + 1, 2 * y, sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        pixel = readPixel(source, 2 * x, 2 * y + 1, sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        pixel = readPixel(source, 2 * x + 1, 2 * y + 1, sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        if (mp == -FLT_MAX) mp = NO_DATA;
    } else {
        mp = FLT_MAX;
        pixel = readPixel(source, 2 * x, 2 * y, sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        pixel = readPixel(source, 2 * x + 1, 2 * y, sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        pixel = readPixel(source, 2 * x, 2 * y + 1, sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        pixel = readPixel(source, 2 * x + 1, 2 * y + 1, sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        if (mp == FLT_MAX)
            mp = NO_DATA;
    }

    dest[y * destWidth + x] = mp;
}

extern "C" void devicePool(int sourceWidth, int sourceHeight, float *source,
                           int destWidth, int destHeight, float *dest,
                           short mode) {
    // Pointers to device memory
    float *deviceSourceImg;
    float *deviceResultImg;

    dim3 blocks(destWidth / POOL_THREAD_DIM + 1, destHeight / POOL_THREAD_DIM + 1);
    dim3 threads(POOL_THREAD_DIM, POOL_THREAD_DIM);

    int srcPixCount = sourceWidth * sourceHeight;
    int destPixCount = destWidth * destHeight;

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg,
                               sizeof(float) * srcPixCount));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg,
                               sizeof(float) * destPixCount));

    checkCudaErrors(cudaMemcpy(deviceSourceImg, source,
        sizeof(float)*srcPixCount, cudaMemcpyHostToDevice));

    pool<<<blocks, threads>>> (sourceWidth, sourceHeight, deviceSourceImg,
            destWidth, destHeight, deviceResultImg, mode);

    checkCudaErrors(cudaMemcpy(dest, deviceResultImg,
        sizeof(float)*destPixCount, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceSourceImg));
    checkCudaErrors(cudaFree(deviceResultImg));
}

/*
 * Uses pooling to extend min/max regions without reducing the resolution
 * of the image.
 */
__global__ void pool_in_place(int width, int height, float *source, float *dest,
                              int radius, short mode) {
    const int x = blockIdx.x * POOL_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * POOL_THREAD_DIM + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float mp = NO_DATA;
    float pixel;

    // Compute the bounds over which to pool.
    int xs = max(x - radius, 0);
    int xe = min(x + radius, width - 1);
    int ys = max(y - radius, 0);
    int ye = min(y + radius, height - 1);

    if (mode == POOL_MAX) {
        mp = -FLT_MAX;
        for (int xi = xs; xi <= xe; ++xi) {
            for (int yi = ys; yi <= ye; ++yi) {
                pixel = source[yi * width + xi];
                mp = (pixel == NO_DATA) ? mp : max(pixel, mp);
            }
        }
        if (mp == -FLT_MAX)
            mp = NO_DATA;
    } else {
        mp = FLT_MAX;
        for (int xi = xs; xi <= xe; ++xi) {
            for (int yi = ys; yi <= ye; ++yi) {
                pixel = source[yi * width + xi];
                mp = (pixel == NO_DATA) ? mp : min(pixel, mp);
            }
        }
        if (mp == FLT_MAX)
            mp = NO_DATA;
    }

    dest[y * width + x] = mp;
}

extern "C" void devicePoolInPlace(int width, int height, float *source, float *dest,
                                  int radius, short mode)
{
    // Pointers to device memory
    float *deviceSourceImg;
    float *deviceResultImg;

    int pixCount = width * height;
    dim3 blocks(width / POOL_THREAD_DIM + 1, height / POOL_THREAD_DIM + 1);
    dim3 threads(POOL_THREAD_DIM, POOL_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg,
                               sizeof(float) * pixCount));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg,
                               sizeof(float) * pixCount));

    // Copy the source image into GPU memory.
    checkCudaErrors(cudaMemcpy(deviceSourceImg, source,
                               sizeof(float)*pixCount,
                               cudaMemcpyHostToDevice));

    pool_in_place<<<blocks, threads>>> (width, height, deviceSourceImg,
                                        deviceResultImg, radius, mode);

    // Copy the final image from GPU memory to dest.
    checkCudaErrors(cudaMemcpy(dest, deviceResultImg,
                               sizeof(float)*pixCount,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceSourceImg));
    checkCudaErrors(cudaFree(deviceResultImg));
}

__global__ void grow_mask(int width, int height, float *source, 
                          float *dest, int steps) {
    const int x = blockIdx.x * POOL_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * POOL_THREAD_DIM + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Get the original pixel value.
    float pixel_val = source[y * width + x];

    // Check each pixel within steps distance.
    int ys = max(0, y - steps);
    int ye = min(height - 1, y + steps);
    for (int yi = ys; yi <= ye; ++yi) {
        int steps_left = steps - abs(y - yi);
        int xs = max(0, x - steps_left);
        int xe = min(width - 1, x + steps_left);
        
        for (int xi = xs; xi <= xe; ++xi){
            if (source[yi * width + xi] == NO_DATA)
                pixel_val = NO_DATA;
        }
    }

    dest[y * width + x] = pixel_val;
}

extern "C" void deviceGrowMask(int width, int height, float *source, 
                               float *dest, int steps) {
    // Pointers to device memory
    float *deviceSourceImg;
    float *deviceResultImg;

    int pixCount = width * height;
    dim3 blocks(width / POOL_THREAD_DIM + 1, height / POOL_THREAD_DIM + 1);
    dim3 threads(POOL_THREAD_DIM, POOL_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg,
                               sizeof(float) * pixCount));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg,
                               sizeof(float) * pixCount));

    // Copy the source image into GPU memory.
    checkCudaErrors(cudaMemcpy(deviceSourceImg, source,
                               sizeof(float)*pixCount,
                               cudaMemcpyHostToDevice));

    grow_mask<<<blocks, threads>>> (width, height, deviceSourceImg,
                                    deviceResultImg, steps);

    // Copy the final image from GPU memory to dest.
    checkCudaErrors(cudaMemcpy(dest, deviceResultImg,
                               sizeof(float)*pixCount,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceSourceImg));
    checkCudaErrors(cudaFree(deviceResultImg));
}


} /* namespace kbmod */

#endif /* IMAGE_KERNELS_CU_ */
