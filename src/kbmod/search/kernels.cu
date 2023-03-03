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

#include "common.h"
#include <cmath>
#include "cuda_errors.h"
#include <stdexcept>
#include <stdio.h>
#include <float.h>

namespace search {

extern "C" __device__ __host__ void sigmaGFilteredIndicesCU(float *values, int num_values, float sGL0,
                                                            float sGL1, float sigmaGCoeff, float width,
                                                            int *idxArray, int *minKeepIndex,
                                                            int *maxKeepIndex) {
    // Clip the percentiles to [0.01, 99.99] to avoid invalid array accesses.
    if (sGL0 < 0.0001) sGL0 = 0.0001;
    if (sGL1 > 0.9999) sGL1 = 0.9999;

    // Initialize the index array.
    for (int j = 0; j < num_values; j++) {
        idxArray[j] = j;
    }

    // Sort the the indexes (idxArray) of values in ascending order.
    int tmpSortIdx;
    for (int j = 0; j < num_values; j++) {
        for (int k = j + 1; k < num_values; k++) {
            if (values[idxArray[j]] > values[idxArray[k]]) {
                tmpSortIdx = idxArray[j];
                idxArray[j] = idxArray[k];
                idxArray[k] = tmpSortIdx;
            }
        }
    }

    // Compute the index of each of the percent values in values
    // from the given bounds sGL0, 0.5 (median), and sGL1.
    const int pct_L = int(ceil(num_values * sGL0) + 0.001) - 1;
    const int pct_H = int(ceil(num_values * sGL1) + 0.001) - 1;
    const int median_ind = int(ceil(num_values * 0.5) + 0.001) - 1;

    // Compute the values that are +/- (width * sigmaG) from the median.
    float sigmaG = sigmaGCoeff * (values[idxArray[pct_H]] - values[idxArray[pct_L]]);
    float minValue = values[idxArray[median_ind]] - width * sigmaG;
    float maxValue = values[idxArray[median_ind]] + width * sigmaG;

    // Find the index of the first value >= minValue.
    int start = 0;
    while ((start < median_ind) && (values[idxArray[start]] < minValue)) {
        ++start;
    }
    *minKeepIndex = start;

    // Find the index of the last value <= maxValue.
    int end = median_ind + 1;
    while ((end < num_values) && (values[idxArray[end]] <= maxValue)) {
        ++end;
    }
    *maxKeepIndex = end - 1;
}

__device__ float readEncodedPixel(void *imageVect, int index, int numBytes, const scaleParameters &params) {
    float value = (numBytes == 1) ? (float)reinterpret_cast<uint8_t *>(imageVect)[index]
                                  : (float)reinterpret_cast<uint16_t *>(imageVect)[index];
    float result = (value == 0.0) ? NO_DATA : (value - 1.0) * params.scale + params.minVal;
    return result;
}

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Returns a
 * fixed number of results per pixel specified by RESULTS_PER_PIXEL
 * filters results using a sigmaG-based filter and a central-moment filter.
 */
__global__ void searchFilterImages(int imageCount, int width, int height, void *psiVect, void *phiVect,
                                   perImageData image_data, searchParameters params, int trajectoryCount,
                                   trajectory *trajectories, trajectory *results) {
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
    const unsigned int pixelsPerImage = width * height;

    // Data structures used for filtering.
    float lcArray[MAX_NUM_IMAGES];
    float psiArray[MAX_NUM_IMAGES];
    float phiArray[MAX_NUM_IMAGES];
    int idxArray[MAX_NUM_IMAGES];

    // Create an initial set of best results with likelihood -1.0.
    // We also set (x, y) because they are used in the later python
    // functions.
    trajectory best[RESULTS_PER_PIXEL];
    for (int r = 0; r < RESULTS_PER_PIXEL; ++r) {
        best[r].x = x;
        best[r].y = y;
        best[r].lh = -1.0;
    }

    // For each trajectory we'd like to search
    for (int t = 0; t < trajectoryCount; ++t) {
        // Create a trajectory for this search.
        trajectory currentT;
        currentT.x = x;
        currentT.y = y;
        currentT.xVel = trajectories[t].xVel;
        currentT.yVel = trajectories[t].yVel;
        currentT.obsCount = 0;

        float psiSum = 0.0;
        float phiSum = 0.0;

        // Loop over each image and sample the appropriate pixel
        for (int i = 0; i < imageCount; ++i) {
            lcArray[i] = 0;
            psiArray[i] = 0;
            phiArray[i] = 0;
            idxArray[i] = i;
        }

        // Loop over each image and sample the appropriate pixel
        int num_seen = 0;
        for (int i = 0; i < imageCount; ++i) {
            // Predict the trajectory's position.
            float cTime = image_data.imageTimes[i];
            int currentX = x + int(currentT.xVel * cTime + 0.5);
            int currentY = y + int(currentT.yVel * cTime + 0.5);

            // If using barycentric correction, apply it.
            // Must be before out of bounds check
            if (params.useCorr && (image_data.baryCorrs != nullptr)) {
                baryCorrection bc = image_data.baryCorrs[i];
                currentX = int(x + currentT.xVel * cTime + bc.dx + x * bc.dxdx + y * bc.dxdy + 0.5);
                currentY = int(y + currentT.yVel * cTime + bc.dy + x * bc.dydx + y * bc.dydy + 0.5);
            }

            // Test if trajectory goes out of the image, in which case we do not
            // look up a pixel value for this time step (allowing trajectories to
            // overlap the image for only some of the times).
            if (currentX >= width || currentY >= height || currentX < 0 || currentY < 0) {
                continue;
            }

            // Get the Psi and Phi pixel values.
            unsigned int pixel_index = (pixelsPerImage * i + currentY * width + currentX);
            float cPsi = (params.psiNumBytes <= 0 || image_data.psiParams == nullptr)
                                 ? reinterpret_cast<float *>(psiVect)[pixel_index]
                                 : readEncodedPixel(psiVect, pixel_index, params.psiNumBytes,
                                                    image_data.psiParams[i]);
            if (cPsi == NO_DATA) continue;

            float cPhi = (params.phiNumBytes <= 0 || image_data.phiParams == nullptr)
                                 ? reinterpret_cast<float *>(phiVect)[pixel_index]
                                 : readEncodedPixel(phiVect, pixel_index, params.phiNumBytes,
                                                    image_data.phiParams[i]);
            if (cPhi == NO_DATA) continue;

            if (cPsi != NO_DATA && cPhi != NO_DATA) {
                currentT.obsCount++;
                psiSum += cPsi;
                phiSum += cPhi;
                psiArray[num_seen] = cPsi;
                phiArray[num_seen] = cPhi;
                if (cPhi != 0.0) lcArray[num_seen] = cPsi / cPhi;
                num_seen += 1;
            }
        }
        currentT.lh = psiSum / sqrt(phiSum);
        currentT.flux = psiSum / phiSum;

        // If we do not have enough observations or a good enough LH score,
        // do not bother with any of the following steps.
        if ((currentT.obsCount < params.minObservations) ||
            (params.do_sigmag_filter && currentT.lh < params.minLH))
            continue;

        // If we are doing on GPU filtering, run the sigmaG filter
        // and recompute the likelihoods.
        if (params.do_sigmag_filter) {
            int minKeepIndex = 0;
            int maxKeepIndex = num_seen - 1;
            sigmaGFilteredIndicesCU(lcArray, num_seen, params.sGL_L, params.sGL_H, params.sigmaGCoeff, 2.0,
                                    idxArray, &minKeepIndex, &maxKeepIndex);

            // Compute the likelihood and flux of the track based on the filtered
            // observations (ones in [minKeepIndex, maxKeepIndex]).
            float newPsiSum = 0.0;
            float newPhiSum = 0.0;
            for (int i = minKeepIndex; i <= maxKeepIndex; i++) {
                int idx = idxArray[i];
                newPsiSum += psiArray[idx];
                newPhiSum += phiArray[idx];
            }
            currentT.lh = newPsiSum / sqrt(newPhiSum);
            currentT.flux = newPsiSum / newPhiSum;
        }

        // Insert the new trajectory into the sorted list of results.
        // Only sort the values with valid likelihoods.
        trajectory temp;
        for (int r = 0; r < RESULTS_PER_PIXEL; ++r) {
            if (currentT.lh > best[r].lh && currentT.lh > -1.0) {
                temp = best[r];
                best[r] = currentT;
                currentT = temp;
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
void *encodeImage(float *imageVect, int numTimes, int numPixels, scaleParameters *params, bool debug) {
    void *deviceVect = NULL;

    long unsigned int total_size = sizeof(T) * numTimes * numPixels;
    if (debug) {
        printf("Encoding image into %lu bytes/pixel for a total of %lu bytes.\n", sizeof(T), total_size);
    }

    // Do the encoding locally first.
    T *encoded = (T *)malloc(total_size);
    for (int t = 0; t < numTimes; ++t) {
        float safe_max = params[t].maxVal - params[t].scale / 100.0;
        for (int p = 0; p < numPixels; ++p) {
            int index = t * numPixels + p;
            float value = imageVect[index];
            if (value == NO_DATA) {
                encoded[index] = 0;
            } else {
                value = min(value, safe_max);
                value = max(value, params[t].minVal);
                value = (value - params[t].minVal) / params[t].scale + 1.0;
                encoded[index] = static_cast<T>(value);
            }
        }
    }

    // Allocate the space on device and do a direct copy.
    checkCudaErrors(cudaMalloc((void **)&deviceVect, total_size));
    checkCudaErrors(cudaMemcpy(deviceVect, encoded, total_size, cudaMemcpyHostToDevice));

    // Free the local space.
    free(encoded);

    return deviceVect;
}

void *encodeImageFloat(float *imageVect, unsigned int vectLength, bool debug) {
    void *deviceVect = NULL;
    long unsigned int total_size = sizeof(float) * vectLength;

    if (debug) {
        printf("Encoding image as float for a total of %lu bytes.\n", total_size);
    }

    checkCudaErrors(cudaMalloc((void **)&deviceVect, total_size));
    checkCudaErrors(cudaMemcpy(deviceVect, imageVect, total_size, cudaMemcpyHostToDevice));
    return deviceVect;
}

extern "C" void deviceSearchFilter(int imageCount, int width, int height, float *psiVect, float *phiVect,
                                   perImageData img_data, searchParameters params, int trajCount,
                                   trajectory *trajectoriesToSearch, int resultsCount,
                                   trajectory *bestTrajects) {
    // Allocate Device memory
    trajectory *deviceTests;
    float *deviceImgTimes;
    void *devicePsi;
    void *devicePhi;
    trajectory *deviceSearchResults;
    baryCorrection *deviceBaryCorrs = nullptr;
    scaleParameters *devicePsiParams = nullptr;
    scaleParameters *devicePhiParams = nullptr;

    // Check the hard coded maximum number of images against the imageCount.
    if (imageCount > MAX_NUM_IMAGES) {
        throw std::runtime_error("Number of images exceeds GPU maximum.");
    }

    if (params.debug) {
        printf("Allocating %lu bytes for testing grid.\n", sizeof(trajectory) * trajCount);
    }
    checkCudaErrors(cudaMalloc((void **)&deviceTests, sizeof(trajectory) * trajCount));

    if (params.debug) {
        printf("Allocating %lu bytes for time data.\n", sizeof(float) * imageCount);
    }
    checkCudaErrors(cudaMalloc((void **)&deviceImgTimes, sizeof(float) * imageCount));

    if (params.debug) {
        printf("Allocating %lu bytes for testing grid.\n", sizeof(trajectory) * trajCount);
    }
    checkCudaErrors(cudaMalloc((void **)&deviceSearchResults, sizeof(trajectory) * resultsCount));

    // Copy trajectories to search
    checkCudaErrors(cudaMemcpy(deviceTests, trajectoriesToSearch, sizeof(trajectory) * trajCount,
                               cudaMemcpyHostToDevice));

    // Copy image times
    checkCudaErrors(cudaMemcpy(deviceImgTimes, img_data.imageTimes, sizeof(float) * imageCount,
                               cudaMemcpyHostToDevice));

    // Copy (and encode) the images. Also copy over the scaling parameters if needed.
    if ((params.psiNumBytes == 1 || params.psiNumBytes == 2) && (img_data.psiParams != nullptr)) {
        checkCudaErrors(cudaMalloc((void **)&devicePsiParams, imageCount * sizeof(scaleParameters)));
        checkCudaErrors(cudaMemcpy(devicePsiParams, img_data.psiParams, imageCount * sizeof(scaleParameters),
                                   cudaMemcpyHostToDevice));
        if (params.psiNumBytes == 1) {
            devicePsi = encodeImage<uint8_t>(psiVect, imageCount, width * height, img_data.psiParams,
                                             params.debug);
        } else {
            devicePsi = encodeImage<uint16_t>(psiVect, imageCount, width * height, img_data.psiParams,
                                              params.debug);
        }
    } else {
        devicePsi = encodeImageFloat(psiVect, imageCount * width * height, params.debug);
    }
    if ((params.phiNumBytes == 1 || params.phiNumBytes == 2) && (img_data.phiParams != nullptr)) {
        checkCudaErrors(cudaMalloc((void **)&devicePhiParams, imageCount * sizeof(scaleParameters)));
        checkCudaErrors(cudaMemcpy(devicePhiParams, img_data.phiParams, imageCount * sizeof(scaleParameters),
                                   cudaMemcpyHostToDevice));
        if (params.phiNumBytes == 1) {
            devicePhi = encodeImage<uint8_t>(phiVect, imageCount, width * height, img_data.phiParams,
                                             params.debug);
        } else {
            devicePhi = encodeImage<uint16_t>(phiVect, imageCount, width * height, img_data.phiParams,
                                              params.debug);
        }
    } else {
        devicePhi = encodeImageFloat(phiVect, imageCount * width * height, params.debug);
    }

    // allocate memory for and copy barycentric corrections
    if (params.useCorr) {
        if (params.debug) {
            printf("Search is using barycentric corrections (%lu bytes).\n",
                   sizeof(baryCorrection) * imageCount);
        }
        checkCudaErrors(cudaMalloc((void **)&deviceBaryCorrs, sizeof(baryCorrection) * imageCount));
        checkCudaErrors(cudaMemcpy(deviceBaryCorrs, img_data.baryCorrs, sizeof(baryCorrection) * imageCount,
                                   cudaMemcpyHostToDevice));
    }

    // Wrap the per-image data into a struct. This struct will be copied by value
    // during the function call, so we don't need to allocate memory for the
    // struct itself. We just set the pointers to the on device vectors.
    perImageData device_image_data;
    device_image_data.numImages = imageCount;
    device_image_data.imageTimes = deviceImgTimes;
    device_image_data.baryCorrs = deviceBaryCorrs;
    device_image_data.psiParams = devicePsiParams;
    device_image_data.phiParams = devicePhiParams;

    // Compute the range of starting pixels to use when setting the blocks and threads.
    // We use the width and height of the search space (as opposed to the image width
    // and height), meaning the blocks/threads will be indexed relative to the search space.
    int search_width = params.x_start_max - params.x_start_min;
    int search_height = params.y_start_max - params.y_start_min;
    dim3 blocks(search_width / THREAD_DIM_X + 1, search_height / THREAD_DIM_Y + 1);
    dim3 threads(THREAD_DIM_X, THREAD_DIM_Y);

    // Launch Search
    searchFilterImages<<<blocks, threads>>>(imageCount, width, height, devicePsi, devicePhi,
                                            device_image_data, params, trajCount, deviceTests,
                                            deviceSearchResults);

    // Read back results
    checkCudaErrors(cudaMemcpy(bestTrajects, deviceSearchResults, sizeof(trajectory) * resultsCount,
                               cudaMemcpyDeviceToHost));

    // Free the on GPU memory.
    if (deviceBaryCorrs != nullptr) checkCudaErrors(cudaFree(deviceBaryCorrs));
    if (devicePhiParams != nullptr) checkCudaErrors(cudaFree(devicePhiParams));
    if (devicePsiParams != nullptr) checkCudaErrors(cudaFree(devicePsiParams));
    checkCudaErrors(cudaFree(devicePhi));
    checkCudaErrors(cudaFree(devicePsi));
    checkCudaErrors(cudaFree(deviceSearchResults));
    checkCudaErrors(cudaFree(deviceImgTimes));
    checkCudaErrors(cudaFree(deviceTests));
}

} /* namespace search */

#endif /* KERNELS_CU_ */
