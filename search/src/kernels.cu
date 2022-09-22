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

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Returns a
 * fixed number of results per pixel specified by RESULTS_PER_PIXEL
 * filters results using a sigmaG-based filter and a central-moment filter.
 */
__global__ void searchFilterImages(int trajectoryCount, int width, int height,
        int imageCount, int minObservations, float *psiPhiImages,
        trajectory *trajectories, trajectory *results, float *imgTimes,
        bool doFilter, float sGL0, float sGL1, float sigmaGCoeff, float minLH,
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

    // Create an initial set of best results with likelihood -1.0.
    trajectory best[RESULTS_PER_PIXEL];
    for (int r=0; r < RESULTS_PER_PIXEL; ++r)
    {
        best[r].lh = -1.0;
    }
    
    // Give up on any trajectories starting outside the image
    if (x >= width || y >= height)
    {
        return;
    }

    const unsigned int pixelsPerImage = width*height;

    // Use a shared array of times that is cached as opposed
    // to constantly reading from global memory.
    __shared__ float sImgTimes[512];
    int idx = threadIdx.x+threadIdx.y*THREAD_DIM_X;
    if (idx<imageCount) sImgTimes[idx] = imgTimes[idx];
    __syncthreads();

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

        // Reset everything a default values.
        for (int i = 0; i < imageCount; ++i)
        {
            lcArray[i] = 0;
            psiArray[i] = 0;
            phiArray[i] = 0;
            idxArray[i] = i;
        }
        
        // Loop over each image and sample the appropriate pixel
        int num_seen = 0;
        for (int i = 0; i < imageCount; ++i)
        {
            // Predict the trajectory's position.
            float cTime = sImgTimes[i];
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
            unsigned int pixel_index = (pixelsPerImage*i + currentY*width
                                        + currentX);
            float2 cPsiPhi = reinterpret_cast<float2*>(psiPhiImages)[pixel_index];

            // Only aggregate the sums and fill in the arrays if
            // we are seeing a non-masked point. Otherwise skip it.
            if (cPsiPhi.x == NO_DATA) continue;

            currentT.obsCount++;
            psiSum += cPsiPhi.x;
            phiSum += cPsiPhi.y;
            psiArray[num_seen] = cPsiPhi.x;
            phiArray[num_seen] = cPsiPhi.y;
            if (cPsiPhi.y == 0.0)
            {
                lcArray[num_seen] = 0.0;
            } else {
                lcArray[num_seen] = cPsiPhi.x/cPsiPhi.y;
            }
            num_seen += 1;
        }
        currentT.lh = psiSum/sqrt(phiSum);
        currentT.flux = psiSum/phiSum;

        // If we don't have enough observations or do not meet the
        // minLH threshold (and are doing filtering) just stop now.
        // It's not worth doing the sigmaG filtering or inserting into
        // the results.
        if ((currentT.obsCount < minObservations) ||
            (doFilter && (currentT.lh < minLH)))
        {
            continue;
        }

        // If we are doing on GPU filtering, run the sigmaG filter
        // and recompute the likelihoods.
        if (doFilter)
        {
            int minKeepIndex = 0;
            int maxKeepIndex = num_seen - 1;
            sigmaGFilteredIndicesCU(lcArray, num_seen, sGL0, sGL1, sigmaGCoeff,
                                    2.0, idxArray, &minKeepIndex, &maxKeepIndex);

            // Compute the likelihood and flux of the track based on the filtered
            // observations (ones in [minKeepIndex, maxKeepIndex]).
            float newPsiSum = 0.0;
            float newPhiSum = 0.0;
            for (int i = minKeepIndex; i <= maxKeepIndex; i++)
            {
                int idx = idxArray[i];
                newPsiSum += psiArray[idx];
                newPhiSum += phiArray[idx];
            }

            // Compute the new likelihood and filter if needed.
            currentT.lh = newPsiSum/sqrt(newPhiSum);
            currentT.flux = newPsiSum/newPhiSum;
        }

        // Insert the new trajectory into the sorted list of results.
        trajectory temp;
        for (int r = 0; r < RESULTS_PER_PIXEL; ++r)
        {
            if (currentT.lh > best[r].lh &&
                currentT.obsCount >= minObservations)
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
deviceSearchFilter(
        int trajCount, int imageCount, int minObservations, int psiPhiSize,
        int resultsCount, trajectory *trajectoriesToSearch, trajectory *bestTrajects,
        float *imageTimes, float *interleavedPsiPhi, int width, int height,
        bool doFilter, float sigmaGLims[2], float sigmaGCoeff, float minLH,
        bool useCorr, baryCorrection *baryCorrs)
{
    // Allocate Device memory
    trajectory *deviceTests;
    float *deviceImgTimes;
    float *devicePsiPhi;
    trajectory *deviceSearchResults;

    checkCudaErrors(cudaMalloc((void **)&deviceTests, sizeof(trajectory)*trajCount));
    checkCudaErrors(cudaMalloc((void **)&deviceImgTimes, sizeof(float)*imageCount));
    checkCudaErrors(cudaMalloc((void **)&devicePsiPhi,
        sizeof(float)*psiPhiSize));
    checkCudaErrors(cudaMalloc((void **)&deviceSearchResults,
        sizeof(trajectory)*resultsCount));

    // Copy trajectories to search
    checkCudaErrors(cudaMemcpy(deviceTests, trajectoriesToSearch,
            sizeof(trajectory)*trajCount, cudaMemcpyHostToDevice));

    // Copy image times
    checkCudaErrors(cudaMemcpy(deviceImgTimes, imageTimes,
            sizeof(float)*imageCount, cudaMemcpyHostToDevice));

    // Copy interleaved buffer of psi and phi images
    checkCudaErrors(cudaMemcpy(devicePsiPhi, interleavedPsiPhi,
        sizeof(float)*psiPhiSize, cudaMemcpyHostToDevice));

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
    searchFilterImages<<<blocks, threads>>> (trajCount, width,
        height, imageCount, minObservations, devicePsiPhi,
        deviceTests, deviceSearchResults, deviceImgTimes, 
        doFilter, sigmaGLims[0], sigmaGLims[1], sigmaGCoeff, minLH, 
        useCorr, deviceBaryCorrs);

    // Read back results
    checkCudaErrors(cudaMemcpy(bestTrajects, deviceSearchResults,
                sizeof(trajectory)*resultsCount, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceTests));
    checkCudaErrors(cudaFree(deviceImgTimes));
    checkCudaErrors(cudaFree(deviceSearchResults));
    checkCudaErrors(cudaFree(devicePsiPhi));

    if (useCorr){
        checkCudaErrors(cudaFree(deviceBaryCorrs));
    }
}

} /* namespace kbmod */

#endif /* KERNELS_CU_ */
