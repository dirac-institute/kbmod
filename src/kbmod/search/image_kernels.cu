/*
 * image_kernels.cu
 *
 * Created on: October 10, 2022
 * (Split from kernels.cu)
 */

#ifndef IMAGE_KERNELS_CU_
#define IMAGE_KERNELS_CU_

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
                     int destWidth, int destHeight, float *dest, short mode,
                     bool two_sided)
{
    const int x = blockIdx.x * POOL_THREAD_DIM + threadIdx.x;
    const int y = blockIdx.y * POOL_THREAD_DIM + threadIdx.y;
    if (x >= destWidth || y >= destHeight)
        return;

    // Compute the inclusive bounds over which to pool.
    int xs = max(0, (two_sided) ? 2 * x - 1 : 2 * x);
    int xe = min(sourceWidth - 1, 2 * x + 1);
    int ys = max(0, (two_sided) ? 2 * y - 1 : 2 * y);
    int ye = min(sourceHeight - 1, 2 * y + 1);

    float mp;
    if (mode == POOL_MAX) {
        mp = -FLT_MAX;
        for (int yi = ys; yi <= ye; ++yi) {
            for (int xi = xs; xi <= xe; ++xi) {
                mp = maxMasked(source[yi * sourceWidth + xi], mp);
            }
        }
        if (mp == -FLT_MAX) mp = NO_DATA;
    } else {
        mp = FLT_MAX;
        for (int yi = ys; yi <= ye; ++yi) {
            for (int xi = xs; xi <= xe; ++xi) {
                mp = minMasked(source[yi * sourceWidth + xi], mp);
            }
        }
        if (mp == FLT_MAX) mp = NO_DATA;
    }

    dest[y * destWidth + x] = mp;
}

extern "C" void devicePool(int sourceWidth, int sourceHeight, float *source,
                           int destWidth, int destHeight, float *dest,
                           short mode, bool two_sided) {
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
            destWidth, destHeight, deviceResultImg, mode, two_sided);

    checkCudaErrors(cudaMemcpy(dest, deviceResultImg,
        sizeof(float)*destPixCount, cudaMemcpyDeviceToHost));

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

extern "C" __device__ __host__ pixelPos findPeakImageVect(int width, int height, float *img,
                                                          bool furthest_from_center) {
    int c_x = width / 2;
    int c_y = height / 2;

    // Initialize the variables for tracking the peak's location.
    pixelPos result = { 0, 0 };
    float max_val = img[0];
    float dist2 = c_x * c_x + c_y * c_y;

    // Search each pixel for the peak.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (img[y * width + x] > max_val) {
                max_val = img[y * width + x];
                result.x = x;
                result.y = y;
                dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
            } else if (img[y * width + x] == max_val) {
                int new_dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
                if ((furthest_from_center && (new_dist2 > dist2)) ||
                    (!furthest_from_center && (new_dist2 < dist2))) {
                    max_val = img[y * width + x];
                    result.x = x;
                    result.y = y;
                    dist2 = new_dist2;
                }
            }
        }
    }

    return result;
}


// Find the basic image moments in order to test if stamps have a gaussian shape.
// It computes the moments on the "normalized" image where the minimum
// value has been shifted to zero and the sum of all elements is 1.0.
// Elements with NO_DATA are treated as zero.
extern "C" __device__ __host__ imageMoments findCentralMomentsImageVect(
            int width, int height, float *img) {
    const int num_pixels = width * height;
    const int c_x = width / 2;
    const int c_y = height / 2;

    // Set all the moments to zero initially.
    imageMoments res = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    // Find the min (non-NO_DATA) value to subtract off.
    float min_val = FLT_MAX;
    for (int p = 0; p < num_pixels; ++p) {
        min_val = ((img[p] != NO_DATA) && (img[p] < min_val)) ? img[p] : min_val;
    }

    // Find the sum of the zero-shifted (non-NO_DATA) pixels.
    double sum = 0.0;
    for (int p = 0; p < num_pixels; ++p) {
        sum += (img[p] != NO_DATA) ? (img[p] - min_val) : 0.0;
    }
    if (sum == 0.0) return res;

    // Compute the rest of the moments.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int ind = y * width + x;
            float pix_val = (img[ind] != NO_DATA) ? (img[ind] - min_val) / sum : 0.0;  

            res.m00 += pix_val;
            res.m10 += (x - c_x) * pix_val;
            res.m20 += (x - c_x) * (x - c_x) * pix_val;
            res.m01 += (y - c_y) * pix_val;
            res.m02 += (y - c_y) * (y - c_y) * pix_val;
            res.m11 += (x - c_x) * (y - c_y) * pix_val;
        }
    }

    return res;
}

__global__ void device_get_coadd_stamp(int num_images, int width, int height, float* image_vect,
                                       perImageData image_data, int radius, bool do_mean,
                                       int num_trajectories, trajectory *trajectories,
                                       int* use_index_vect, float* results) {
    // Get the trajectory that we are going to be using.
    const int trj_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (trj_index < 0 || trj_index >= num_trajectories)
        return;
    trajectory trj = trajectories[trj_index];
    int use_index_offset = num_images * trj_index;

    // Allocate space for the coadd information and initialize to zero.
    const int pixels_per_image = width * height;
    const int stamp_width = 2 * radius + 1;
    const int stamp_ppi = stamp_width * stamp_width;
    float sum[MAX_STAMP_EDGE * MAX_STAMP_EDGE];
    float count[MAX_STAMP_EDGE * MAX_STAMP_EDGE];
    for (int i = 0; i < stamp_ppi; ++i) {
        sum[i] = 0.0;
        count[i] = 0.0;
    }

    // Loop over each image and compute the stamp.
    for (int t = 0; t < num_images; ++t) {
        // Skip entries marked 0 in the use_index_vect.
        if (use_index_vect != nullptr && use_index_vect[use_index_offset + t] == 0) {
            continue;
        }

        // Predict the trajectory's position including the barycentric correction if needed.
        float cTime = image_data.imageTimes[t];
        int currentX = trj.x + int(trj.xVel * cTime + 0.5);
        int currentY = trj.y + int(trj.yVel * cTime + 0.5);
        if (image_data.baryCorrs != nullptr) {
            baryCorrection bc = image_data.baryCorrs[t];
            currentX = int(trj.x + trj.xVel*cTime + bc.dx + trj.x*bc.dxdx + trj.y*bc.dxdy + 0.5);
            currentY = int(trj.y + trj.yVel*cTime + bc.dy + trj.x*bc.dydx + trj.y*bc.dydy + 0.5);
        }

        // Get the stamp and add it to the running totals..
        for (int stamp_y = 0; stamp_y < stamp_width; ++stamp_y) {
            int img_y = currentY - radius + stamp_y;
            for (int stamp_x = 0; stamp_x < stamp_width; ++stamp_x) {
                int img_x = currentX - radius + stamp_x;
                if ((img_x >= 0) && (img_x < width) && (img_y >= 0) && (img_y < height)) {
                    int pixel_index = pixels_per_image * t + img_y * width + img_x;
                    if (image_vect[pixel_index] != NO_DATA) {
                        int stamp_index = stamp_y * stamp_width + stamp_x;
                        sum[stamp_index] += image_vect[pixel_index];
                        count[stamp_index] += 1.0;
                    }
                }
            }
        }    
    }

    // Compute the mean if needed.
    if (do_mean) {
        for (int i = 0; i < stamp_ppi; ++i) {
            sum[i] = (count[i] > 0.0) ? sum[i]/count[i] : 0.0;
        }
    }

    // Save the result.
    int offset = stamp_ppi * trj_index;
    for (int i = 0; i < stamp_ppi; ++i) {
        results[offset + i] = sum[i];
    }
}

void deviceGetCoadds(ImageStack& stack, perImageData image_data, int radius, bool do_mean,
                     int num_trajectories, trajectory *trajectories,
                     std::vector<std::vector<bool> >& use_index_vect, float* results) {
    // Allocate Device memory
    trajectory *device_trjs;
    int *device_use_index = nullptr;
    float *device_times;
    float *device_img;
    float *device_res;
    baryCorrection* deviceBaryCorrs = nullptr;

    // Compute the dimensions for the data.
    const unsigned int num_images = stack.imgCount();
    const unsigned int width = stack.getWidth();
    const unsigned int height = stack.getHeight();
    const unsigned int num_image_pixels = num_images * width * height;
    const unsigned int stamp_ppi = (2 * radius + 1) * (2 * radius + 1);
    const unsigned int num_stamp_pixels = num_trajectories * stamp_ppi;

    // Allocate and copy the trajectories.
    checkCudaErrors(cudaMalloc((void **)&device_trjs, sizeof(trajectory) * num_trajectories));
    checkCudaErrors(cudaMemcpy(device_trjs, trajectories, sizeof(trajectory) * num_trajectories,
                               cudaMemcpyHostToDevice));

    // Check if we need to create a vector of per-trajectory, per-image use.
    // Convert the vector of booleans into an integer array so we do a cudaMemcpy.
    if (use_index_vect.size() == num_trajectories) {
        checkCudaErrors(cudaMalloc((void **)&device_use_index,
                                   sizeof(int) * num_images * num_trajectories));

        int* start_ptr = device_use_index;
        std::vector<int> int_vect(num_images, 0);
        for (unsigned i = 0; i < num_trajectories; ++i) {
            assert(use_index_vect[i].size() == num_images);
            for (unsigned t = 0; t < num_images; ++t) {
                int_vect[t] = use_index_vect[i][t] ? 1 : 0;
            }
            
            checkCudaErrors(cudaMemcpy(start_ptr, int_vect.data(), 
                                       sizeof(int) * num_images,
                                       cudaMemcpyHostToDevice));
            start_ptr += num_images;
        }
    }

    // Allocate and copy the times.
    checkCudaErrors(cudaMalloc((void **)&device_times, sizeof(float) * num_images));
    checkCudaErrors(cudaMemcpy(device_times, image_data.imageTimes,
            sizeof(float) * num_images, cudaMemcpyHostToDevice));

    // Allocate and copy the images.
    checkCudaErrors(cudaMalloc((void **)&device_img, sizeof(float) * num_image_pixels));
    float* next_ptr = device_img;
    for (unsigned t = 0; t < num_images; ++t) {
        const std::vector<float>& data_ref = stack.getSingleImage(t).getScience().getPixels();  

        assert(data_ref.size() == width * height);
        checkCudaErrors(cudaMemcpy(next_ptr, data_ref.data(), sizeof(float) * width * height,
                                   cudaMemcpyHostToDevice));
        next_ptr += width * height;
    }

    // Allocate space for the results.
    checkCudaErrors(cudaMalloc((void **)&device_res, sizeof(float) * num_stamp_pixels));

    // Allocate memory for and copy barycentric corrections (if needed).
    if (image_data.baryCorrs != nullptr) {
        checkCudaErrors(cudaMalloc((void **)&deviceBaryCorrs,
            sizeof(baryCorrection) * num_images));
        checkCudaErrors(cudaMemcpy(deviceBaryCorrs, image_data.baryCorrs,
            sizeof(baryCorrection) * num_images, cudaMemcpyHostToDevice));
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

    dim3 blocks(num_trajectories / 256 + 1);
    dim3 threads(256);

    // Launch Search
    device_get_coadd_stamp<<<blocks, threads>>> (num_images, width, height, device_img,
                                                 device_image_data, radius, do_mean,
                                                 num_trajectories, device_trjs, device_use_index,
                                                 device_res);

    // Read back results
    checkCudaErrors(cudaMemcpy(results, device_res, sizeof(float) * num_stamp_pixels,
                               cudaMemcpyDeviceToHost));

    // Free the on GPU memory.
    if (deviceBaryCorrs != nullptr)
        checkCudaErrors(cudaFree(deviceBaryCorrs));
    checkCudaErrors(cudaFree(device_res));
    checkCudaErrors(cudaFree(device_img));
    if (device_use_index != nullptr)
        checkCudaErrors(cudaFree(device_use_index));
    checkCudaErrors(cudaFree(device_times));
    checkCudaErrors(cudaFree(device_trjs));
}


} /* namespace search */

#endif /* IMAGE_KERNELS_CU_ */
