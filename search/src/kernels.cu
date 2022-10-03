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
#include <helper_cuda.h>
#include <stdio.h>
#include <float.h>
#include "filtering_kernels.cu"

namespace kbmod {


/*
 * Device kernel that convolves the provided image with the psf
 */
__global__ void convolvePSF(int width, int height,
    float *sourceImage, float *resultImage, float *psf,
    int psfRad, int psfDim, float psfSum, float maskFlag)
{
    // Find bounds of convolution area
    const int x = blockIdx.x*CONV_THREAD_DIM+threadIdx.x;
    const int y = blockIdx.y*CONV_THREAD_DIM+threadIdx.y;
    if (x < 0 || x > width-1 || y < 0 || y > height-1) return;

    // Read kernel
    float sum = 0.0;
    float psfPortion = 0.0;
    float center = sourceImage[y*width+x];
    if (center != NO_DATA) {
        for (int j = -psfRad; j <= psfRad; j++)
        {
            // #pragma unroll
            for (int i = -psfRad; i <= psfRad; i++)
            {
                if ((x + i >= 0) && (x + i < width) &&
                    (y + j >= 0) && (y + j < height))
                {
                    float currentPixel = sourceImage[(y+j)*width+(x+i)];
                    if (currentPixel != NO_DATA)
                    {
                        float currentPSF = psf[(j+psfRad)*psfDim+(i+psfRad)];
                        psfPortion += currentPSF;
                        sum += currentPixel * currentPSF;
                    }
                }
            }
        }

        resultImage[y*width+x] = (sum*psfSum)/psfPortion;
    } else {
        // Leave masked pixel alone (these could be replaced here with zero)
        resultImage[y*width+x] = NO_DATA; // 0.0
    }
}

extern "C" void deviceConvolve(float *sourceImg, float *resultImg,
    int width, int height, float *psfKernel,
    int psfSize, int psfDim, int psfRadius, float psfSum)
{
    // Pointers to device memory //
    float *deviceKernel;
    float *deviceSourceImg;
    float *deviceResultImg;

    long pixelsPerImage = width*height;
    dim3 blocks(width/CONV_THREAD_DIM+1,height/CONV_THREAD_DIM+1);
    dim3 threads(CONV_THREAD_DIM,CONV_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceKernel, sizeof(float)*psfSize));
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg, sizeof(float)*pixelsPerImage));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg, sizeof(float)*pixelsPerImage));

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
__device__ float readPixel(float* img, int x, int y, int width, int height)
{
    return (x<width && y<height) ? img[y*width+x] : NO_DATA;
}

__device__ float maxMasked(float pixel, float previousMax)
{
    return pixel == NO_DATA ? previousMax : max(pixel, previousMax);
}

__device__ float minMasked(float pixel, float previousMin)
{
    return pixel == NO_DATA ? previousMin : min(pixel, previousMin);
}

/*
 * Reduces the resolution of an image to 1/4 using max pooling
 */
__global__ void pool(int sourceWidth, int sourceHeight, float *source,
    int destWidth, int destHeight, float *dest, short mode)
{
    const int x = blockIdx.x*POOL_THREAD_DIM+threadIdx.x;
    const int y = blockIdx.y*POOL_THREAD_DIM+threadIdx.y;
    if (x>=destWidth || y>=destHeight) return;
    float mp;
    float pixel;
    if (mode == POOL_MAX) {
        mp = -FLT_MAX;
        pixel = readPixel(source, 2*x,   2*y,   sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        pixel = readPixel(source, 2*x+1, 2*y,   sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        pixel = readPixel(source, 2*x,   2*y+1, sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        pixel = readPixel(source, 2*x+1, 2*y+1, sourceWidth, sourceHeight);
        mp = maxMasked(pixel, mp);
        if (mp == -FLT_MAX) mp = NO_DATA;
    } else {
        mp = FLT_MAX;
        pixel = readPixel(source, 2*x,   2*y,   sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        pixel = readPixel(source, 2*x+1, 2*y,   sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        pixel = readPixel(source, 2*x,   2*y+1, sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        pixel = readPixel(source, 2*x+1, 2*y+1, sourceWidth, sourceHeight);
        mp = minMasked(pixel, mp);
        if (mp == FLT_MAX) mp = NO_DATA;
    }

    dest[y*destWidth+x] = mp;
}

extern "C" void devicePool(int sourceWidth, int sourceHeight, float *source,
    int destWidth, int destHeight, float *dest, short mode)
{
    // Pointers to device memory //
    float *deviceSourceImg;
    float *deviceResultImg;

    dim3 blocks(destWidth/POOL_THREAD_DIM+1,destHeight/POOL_THREAD_DIM+1);
    dim3 threads(POOL_THREAD_DIM,POOL_THREAD_DIM);

    int srcPixCount = sourceWidth*sourceHeight;
    int destPixCount = destWidth*destHeight;

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg, sizeof(float)*srcPixCount));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg, sizeof(float)*destPixCount));

    checkCudaErrors(cudaMemcpy(deviceSourceImg, source,
        sizeof(float)*srcPixCount, cudaMemcpyHostToDevice));

    pool<<<blocks, threads>>> (sourceWidth, sourceHeight, deviceSourceImg,
            destWidth, destHeight, deviceResultImg, mode);

    checkCudaErrors(cudaMemcpy(dest, deviceResultImg,
        sizeof(float)*destPixCount, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceSourceImg));
    checkCudaErrors(cudaFree(deviceResultImg));
}



extern "C" void* encodeImage(float *imageVect, unsigned int vectLength, 
                             int numBytes, float minVal, float maxVal, float scale)
{
    void* deviceVect = NULL;

    if (numBytes < 1)
    {
        // Allocate space for the array of floats. 
        checkCudaErrors(cudaMalloc((void **)&deviceVect,
                                   sizeof(float)*vectLength));

        // Copy the image vector.
        checkCudaErrors(cudaMemcpy(deviceVect, imageVect,
                                   sizeof(float)*vectLength,
                                   cudaMemcpyHostToDevice));
    } else if (numBytes == 1)
    {
        // Use a maximum value that is slightly smaller than maxVal
        // but in the same bucket.
        float safe_max = maxVal - scale/10.0;

        // Do the encoding on the host first.
        unsigned int total_size = sizeof(uint8_t) * vectLength;
        uint8_t* encoded = (uint8_t*)malloc(total_size);
        for (unsigned int i = 0; i < vectLength; ++i)
        {
            float value = imageVect[i];
            if (value == NO_DATA)
            {
                encoded[i] = 0;
            } else {
                value = min(value, safe_max);
                value = max(value, minVal);
                value = (value - minVal) / scale;
                encoded[i] = (uint8_t)(value) + 1;
            }
        }
        
        // Allocate the space on device and do a direct copy.
        checkCudaErrors(cudaMalloc((void **)&deviceVect, total_size));
        checkCudaErrors(cudaMemcpy(deviceVect, encoded, total_size,
                                   cudaMemcpyHostToDevice));

        // Free the local space.
        free(encoded);
    } else if (numBytes == 2)
    {
        // Use a maximum value that is slightly smaller than maxVal
        // but in the same bucket.
        float safe_max = maxVal - scale/10.0;

        // Do the encoding on the host first.
        unsigned int total_size = sizeof(uint16_t) * vectLength;
        uint16_t* encoded = (uint16_t*)malloc(total_size);
        for (unsigned int i = 0; i < vectLength; ++i)
        {
            float value = imageVect[i];
            if (value == NO_DATA)
            {
                encoded[i] = 0;
            } else {
                value = min(value, safe_max);
                value = max(value, minVal);
                value = (value - minVal) / scale;
                encoded[i] = (uint16_t)(value) + 1;
            }
        }
        
        // Allocate the space on device and do a direct copy.
        checkCudaErrors(cudaMalloc((void **)&deviceVect, total_size));
        checkCudaErrors(cudaMemcpy(deviceVect, encoded, total_size,
                                   cudaMemcpyHostToDevice));

        // Free the local space.
        free(encoded);
    }

    return deviceVect;
}

__device__ float readEncodedPixel(void* imageVect, int index, int numBytes,
                                  float minVal, float scale)
{
    float value = (numBytes == 1) ? 
            (float)reinterpret_cast<uint8_t*>(imageVect)[index] :
            (float)reinterpret_cast<uint16_t*>(imageVect)[index];
    float result = (value == 0.0) ? NO_DATA : (value - 1.0) * scale + minVal;
    return result;
}

/*
 * Uses pooling to extend min/max regions without reducing the resolution
 * of the image.
 */
__global__ void pool_in_place(int width, int height, float *source, float *dest,
                              int radius, short mode)
{
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

    if (mode == POOL_MAX) 
    {
        mp = -FLT_MAX;
        for (int xi = xs; xi <= xe; ++xi)
        {
            for (int yi = ys; yi <= ye; ++yi)
            {
                pixel = source[yi * width + xi];
                mp = (pixel == NO_DATA) ? mp : max(pixel, mp);
            }
        }
        if (mp == -FLT_MAX) mp = NO_DATA;
    } else {
        mp = FLT_MAX;
        for (int xi = xs; xi <= xe; ++xi)
        {
            for (int yi = ys; yi <= ye; ++yi)
            {
                pixel = source[yi * width + xi];
                mp = (pixel == NO_DATA) ? mp : min(pixel, mp);
            }
        }
        if (mp == FLT_MAX) mp = NO_DATA;
    }

    dest[y * width + x] = mp;
}

extern "C" void devicePoolInPlace(int width, int height, float *source, float *dest,
                                  int radius, short mode)
{
    // Pointers to device memory //
    float *deviceSourceImg;
    float *deviceResultImg;

    int pixCount = width * height;
    dim3 blocks(width/POOL_THREAD_DIM + 1, height/POOL_THREAD_DIM + 1);
    dim3 threads(POOL_THREAD_DIM, POOL_THREAD_DIM);

    // Allocate Device memory
    checkCudaErrors(cudaMalloc((void **)&deviceSourceImg, sizeof(float)*pixCount));
    checkCudaErrors(cudaMalloc((void **)&deviceResultImg, sizeof(float)*pixCount));

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

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Returns a
 * fixed number of results per pixel specified by RESULTS_PER_PIXEL
 * filters results using a sigmaG-based filter and a central-moment filter.
 */ 
__global__ void searchFilterImages(int imageCount, int width, int height,
        void *psiVect, void* phiVect, float *imageTimes,
        searchParameters* params, int trajectoryCount,
        trajectory *trajectories, trajectory *results,
        bool useCorr, baryCorrection *baryCorrs)
{
    // Get origin pixel for the trajectories.
    const unsigned short x = blockIdx.x*THREAD_DIM_X+threadIdx.x;
    const unsigned short y = blockIdx.y*THREAD_DIM_Y+threadIdx.y;

    // Data structures used for filtering.
    float lcArray[MAX_NUM_IMAGES];
    float psiArray[MAX_NUM_IMAGES];
    float phiArray[MAX_NUM_IMAGES];
    int idxArray[MAX_NUM_IMAGES];
    int tmpSortIdx;

    // Create an initial set of best results with likelihood -1.0.
    // We also set (x, y) because they are used in the later python
    // functions.
    trajectory best[RESULTS_PER_PIXEL];
    for (int r = 0; r < RESULTS_PER_PIXEL; ++r)
    {
        best[r].x = x;
        best[r].y = y;
        best[r].lh = -1.0;
    }
    
    // Give up on any trajectories starting outside the image
    if (x >= width || y >= height)
    {
        return;
    }

    const unsigned int pixelsPerImage = width * height;

    // For each trajectory we'd like to search
    for (int t=0; t < trajectoryCount; ++t)
    {
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
        for (int i = 0; i < imageCount; ++i)
        {
            lcArray[i] = 0;
            psiArray[i] = 0;
            phiArray[i] = 0;
            idxArray[i] = i;

            // Predict the trajectory's position.
            float cTime = imageTimes[i];
            int currentX = x + int(currentT.xVel*cTime+0.5);
            int currentY = y + int(currentT.yVel*cTime+0.5);

            // If using barycentric correction, apply it
            // This branch is short, and all threads should
            // have same value of baryCorr, so hopefully
            // performance is OK?
            // Must be before out of bounds check
            if (useCorr) {
                baryCorrection bc = baryCorrs[i];
                currentX = int(x + currentT.xVel*cTime + bc.dx + x*bc.dxdx + y*bc.dxdy + 0.5);
                currentY = int(y + currentT.yVel*cTime + bc.dy + x*bc.dydx + y*bc.dydy + 0.5);
            }
                
            // Test if trajectory goes out of image bounds
            // Branching could be avoided here by setting a
            // black image border and clamping coordinates
            if (currentX >= width || currentY >= height
                || currentX < 0 || currentY < 0)
            {
                continue;
            }

            // Get the Psi and Phi pixel values.
            unsigned int pixel_index = (pixelsPerImage*i + currentY*width + currentX);
            float cPsi = (!params->encodeImg) ? reinterpret_cast<float*>(psiVect)[pixel_index] :
                             readEncodedPixel(psiVect, pixel_index, params->psiNumBytes,
                                              params->minPsiVal, params->psiScale);
            if (cPsi == NO_DATA) continue;
            float cPhi = (!params->encodeImg) ? reinterpret_cast<float*>(phiVect)[pixel_index] :
                             readEncodedPixel(phiVect, pixel_index, params->phiNumBytes,
                                              params->minPhiVal, params->phiScale);

            currentT.obsCount++;
            psiSum += cPsi;
            phiSum += cPhi;
            psiArray[i] = cPsi;
            phiArray[i] = cPhi;
            if (cPhi == 0.0)
            {
                lcArray[i] = 0;
            } else {
                lcArray[i] = cPsi/cPhi;
            }
        }
        currentT.lh = psiSum/sqrt(phiSum);
        currentT.flux = psiSum/phiSum;

        // If we do not have enough observations or a good enough LH score,
        // do not bother with any of the following steps.
        if ((currentT.obsCount < minObservations) || 
            (params->doFilter && currentT.lh < params->minLH))
            continue;

        if (doFilter)
        {
            // Sort the the indexes (idxArray) of lcArray in ascending order.
            for (int j = 0; j < imageCount; j++)
            {
                for (int k = j+1; k < imageCount; k++)
                {
                     if (lcArray[idxArray[j]] > lcArray[idxArray[k]])
                     {
                         tmpSortIdx = idxArray[j];
                         idxArray[j] = idxArray[k];
                         idxArray[k] = tmpSortIdx;
                     }
                }
            }

            // Compute index of the three percentile values in lcArray
            // from the given bounds sGL0, 0.5 (median), and sGL1.
            int minKeepIndex = 0;
            int maxKeepIndex = imageCount - 1;
            int imgCountPlus1 = imageCount + 1;
            const int percentiles[3] = {
                int(imgCountPlus1 * params->sGL_L + 0.5) - 1,
                int(imgCountPlus1 * 0.5 + 0.5) - 1,
                int(imgCountPlus1 * params->sGL_H + 0.5) - 1};

            // Compute the lcValues that at +/- 2*sigmaG from the median.
            // This will be used to filter anything outside that range.
            float sigmaG = params->sigmaGCoeff * (lcArray[idxArray[percentiles[2]]]
                               - lcArray[idxArray[percentiles[0]]]);
            float minValue = lcArray[idxArray[percentiles[1]]] - 2 * sigmaG;
            float maxValue = lcArray[idxArray[percentiles[1]]] + 2 * sigmaG;

            // Find the index of the first value in lcArray greater
            // than or equal to minValue.
            for (int i = 0; i <= percentiles[1]; i++)
            {
                int idx = idxArray[i];
                if (lcArray[idx] >= minValue)
                {
                    minKeepIndex = i;
                    break;
                }
            }
            
            // Find the index of the last value in lcArray less
            // than or equal to maxValue.
            for (int i = percentiles[1] + 1; i < imageCount; i++)
            {
                int idx = idxArray[i];
                if (lcArray[idx] <= maxValue)
                {
                    maxKeepIndex = i;
                } else {
                    break;
                }
            }
            
            // Compute the likelihood and flux of the track based on the filtered
            // observations (ones with minValue <= lc <= maxValue).
            float newPsiSum = 0.0;
            float newPhiSum = 0.0;
            for (int i = minKeepIndex; i < maxKeepIndex + 1; i++)
            {
                int idx = idxArray[i];
                newPsiSum += psiArray[idx];
                newPhiSum += phiArray[idx];
            }
            currentT.lh = newPsiSum/sqrt(newPhiSum);
            currentT.flux = newPsiSum/newPhiSum;
        }

        // Insert the new trajectory into the sorted list of results.
        // Only sort the values with valid likelihoods.
        trajectory temp;
        for (int r = 0; r < RESULTS_PER_PIXEL; ++r)
        {
            if (currentT.lh > best[r].lh &&
                currentT.lh > -1.0)
            {
                temp = best[r];
                best[r] = currentT;
                currentT = temp;
            }
        }
    }
    
    // Copy the sorted list of best results for this pixel into
    // the correct location within the global results vector.
    const int base_index = (y * width + x) * RESULTS_PER_PIXEL;
    for (int r = 0; r < RESULTS_PER_PIXEL; ++r)
    {
        results[base_index + r] = best[r];
    }
}

extern "C" void
deviceSearchFilter(int imageCount, int width, int height,
                   float *psiVect, float* phiVect, float *imageTimes,
                   searchParameters* params,
                   int trajCount, trajectory *trajectoriesToSearch,
                   int resultsCount, trajectory *bestTrajects,
                   bool useCorr, baryCorrection *baryCorrs)
{
    // Allocate Device memory
    trajectory *deviceTests;
    float *deviceImgTimes;
    void *devicePsi;
    void *devicePhi;
    trajectory *deviceSearchResults;
    searchParameters *deviceParams;

    checkCudaErrors(cudaMalloc((void **)&deviceParams, sizeof(searchParameters)));
    checkCudaErrors(cudaMalloc((void **)&deviceTests, sizeof(trajectory)*trajCount));
    checkCudaErrors(cudaMalloc((void **)&deviceImgTimes, sizeof(float)*imageCount));
    checkCudaErrors(cudaMalloc((void **)&deviceSearchResults,
        sizeof(trajectory)*resultsCount));

    // Copy the Paramter settings.
    checkCudaErrors(cudaMemcpy(deviceParams, params,
            sizeof(searchParameters), cudaMemcpyHostToDevice));

    // Copy trajectories to search
    checkCudaErrors(cudaMemcpy(deviceTests, trajectoriesToSearch,
            sizeof(trajectory)*trajCount, cudaMemcpyHostToDevice));

    // Copy image times
    checkCudaErrors(cudaMemcpy(deviceImgTimes, imageTimes,
            sizeof(float)*imageCount, cudaMemcpyHostToDevice));

    // Copy (and encode) the images.
    unsigned int vectLength = imageCount * width * height;
    devicePsi = encodeImage(psiVect, vectLength, params->encodeImg ? params->psiNumBytes : -1,
                            params->minPsiVal, params->maxPsiVal, params->psiScale);
    devicePhi = encodeImage(phiVect, vectLength, params->encodeImg ? params->phiNumBytes : -1,
                            params->minPhiVal, params->maxPhiVal, params->phiScale);

    // allocate memory for and copy barycentric corrections
    baryCorrection* deviceBaryCorrs;
    if (useCorr) {
        checkCudaErrors(cudaMalloc((void **)&deviceBaryCorrs,
            sizeof(baryCorrection)*imageCount));
        checkCudaErrors(cudaMemcpy(deviceBaryCorrs, baryCorrs,
            sizeof(baryCorrection)*imageCount, cudaMemcpyHostToDevice));
    }

    dim3 blocks(width/THREAD_DIM_X+1,height/THREAD_DIM_Y+1);
    dim3 threads(THREAD_DIM_X,THREAD_DIM_Y);

    // Launch Search
    searchFilterImages<<<blocks, threads>>> (imageCount, width, height,
         devicePsi, devicePhi, deviceImgTimes, deviceParams,
         trajCount, deviceTests, deviceSearchResults,
         useCorr, deviceBaryCorrs);

    // Read back results
    checkCudaErrors(cudaMemcpy(bestTrajects, deviceSearchResults,
                sizeof(trajectory)*resultsCount, cudaMemcpyDeviceToHost));

    // Free the on GPU memory.
    checkCudaErrors(cudaFree(deviceTests));
    checkCudaErrors(cudaFree(deviceImgTimes));
    checkCudaErrors(cudaFree(deviceSearchResults));
    checkCudaErrors(cudaFree(deviceParams));
    checkCudaErrors(cudaFree(devicePsi));
    checkCudaErrors(cudaFree(devicePhi));
    if (useCorr){
        checkCudaErrors(cudaFree(deviceBaryCorrs));
    }
}

} /* namespace kbmod */

#endif /* KERNELS_CU_ */
