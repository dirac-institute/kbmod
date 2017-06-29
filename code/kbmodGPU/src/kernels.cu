/*
 * kernels.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: kbmod-usr
 */

#ifndef KERNELS_CU_
#define KERNELS_CU_

#include "common.h"
#include "PointSpreadFunc.h"
#include <helper_cuda.h>


/*
 * Device kernel that convolves the provided image with the psf
 */
__global__ void convolvePSF(int width, int height,
	float *sourceImage, float *resultImage, float *psf,
	int psfRad, int psfDim, float psfSum, float maskFlag)
{
	// Find bounds of convolution area
	const int x = blockIdx.x*32+threadIdx.x;
	const int y = blockIdx.y*32+threadIdx.y;
	if (x < 0 || x > width-1 || y < 0 || y > height-1) return;
	const int minX = max(x-psfRad, 0);
	const int minY = max(y-psfRad, 0);
	const int maxX = min(x+psfRad, width-1);
	const int maxY = min(y+psfRad, height-1);
	const int dx = maxX-minX;
	const int dy = maxY-minY;
	if (dx < 1 || dy < 1 ) return;

	// Read kernel
	float sum = 0.0;
	float psfPortion = 0.0;
	float center = sourceImage[y*width+x];
	if (center > MASK_FLAG/2 /*!= MASK_FLAG*/) {
		for (int j=minY; j<=maxY; ++j)
		{
			// #pragma unroll
			for (int i=minX; i<=maxX; ++i)
			{
				float currentPixel = sourceImage[j*width+i];
				if (currentPixel > MASK_FLAG/2 /*!= MASK_FLAG*/) {
					float currentPSF = psf[(j-minY)*psfDim+i-minX];
					psfPortion += currentPSF;
					sum += currentPixel * currentPSF;
				}
			}
		}

		resultImage[y*width+x] = (sum*psfSum)/psfPortion;
	} else {
		resultImage[y*width+x] = center;
	}
}

extern "C" void deviceConvolve(float *sourceImg, float *resultImg,
long *dimensions, PointSpreadFunc *PSF)
{
	// Pointers to device memory //
	float *deviceKernel;
	float *deviceSourceImg;
	float *deviceResultImg;

	long pixelsPerImage = dimensions[0]*dimensions[1];
	dim3 blocks(dimensions[0]/32+1,dimensions[1]/32+1);
	dim3 threads(32,32);

	// Allocate Device memory
	checkCudaErrors(cudaMalloc((void **)&deviceKernel, sizeof(float)*PSF->getSize()));
	checkCudaErrors(cudaMalloc((void **)&deviceSourceImg, sizeof(float)*pixelsPerImage));
	checkCudaErrors(cudaMalloc((void **)&deviceResultImg, sizeof(float)*pixelsPerImage));

	checkCudaErrors(cudaMemcpy(deviceKernel, PSF->kernelData(),
		sizeof(float)*PSF->getSize(), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceSourceImg, sourceImg,
		sizeof(float)*pixelsPerImage, cudaMemcpyHostToDevice));

	convolvePSF<<<blocks, threads>>> (dimensions[0], dimensions[1], deviceSourceImg,
		deviceResultImg, deviceKernel, PSF->getRadius(), PSF->getDim(), PSF->getSum(), MASK_FLAG);

	checkCudaErrors(cudaMemcpy(resultImg, deviceResultImg,
		sizeof(float)*pixelsPerImage, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(deviceKernel));
	checkCudaErrors(cudaFree(deviceSourceImg));
	checkCudaErrors(cudaFree(deviceResultImg));

}

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Returns a
 * fixed number of results per pixel specified by RESULTS_PER_PIXEL
 */
__global__ void searchImages(int trajectoryCount, int width,
	int height, int imageCount, float *psiPhiImages,
	trajectory *trajectories, trajectory *results, float *imgTimes
	/*float slopeRejectThresh, float fluxPix, float termThreshold*/)
{

	// Get trajectory origin
	const unsigned short x = blockIdx.x*THREAD_DIM_X+threadIdx.x;
	const unsigned short y = blockIdx.y*THREAD_DIM_Y+threadIdx.y;

	trajectory best[RESULTS_PER_PIXEL];
	for (int r=0; r<RESULTS_PER_PIXEL; ++r)
	{
		best[r]  = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0,
		.flux = 0.0, .x = x, .y = y };
	}

	__shared__ float sImgTimes[256];
	int idx = threadIdx.x*THREAD_DIM_X+threadIdx.y;
	if (idx<imageCount) sImgTimes[idx] = imgTimes[idx];

	// Give up on any trajectories starting outside the image
	if (x >= width || y >= height)
	{
		return;
	}

	const unsigned int pixelsPerImage = width*height;

	// For each trajectory we'd like to search
	for (int t=0; t<trajectoryCount; ++t)
	{
	  	trajectory currentT = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0,
		.flux = 0.0, .x = x, .y = y };

		currentT.xVel = trajectories[t].xVel;
		currentT.yVel = trajectories[t].yVel;
		float psiSum = 0.0;
		float phiSum = 0.0;

		// Loop over each image and sample the appropriate pixel
		for (int i=0; i<imageCount; ++i)
		{
			float cTime = sImgTimes[i];
			int currentX = x + int(currentT.xVel*cTime+0.5);
			int currentY = y + int(currentT.yVel*cTime+0.5);
			// Test if trajectory goes out of image bounds
			// Branching could be avoided here by setting a
			// black image border and clamping coordinates
			if (currentX >= width || currentY >= height
			    || currentX < 0 || currentY < 0)
			{
				// Penalize trajctories that leave edge
				//psiSum += -0.1;
				continue;
			}
			int pixel = (pixelsPerImage*i +
				 currentY*width +
				 currentX);

			//float cPsi = psiPhiImages[pixel];
			//float cPhi = psiPhiImages[pixel+1];
			float2 cPsiPhi = reinterpret_cast<float2*>(psiPhiImages)[pixel];

			psiSum += cPsiPhi.x;// < MASK_FLAG/2 /*== MASK_FLAG*/ ? 0.0 : cPsiPhi.x;//min(cPsi,0.3);
			phiSum += cPsiPhi.y;// < MASK_FLAG/2 /*== MASK_FLAG*/ ? 0.0 : cPsiPhi.y;

			//if (psiSum <= 0.0 && i>4) break;
		}

		// Just in case a phiSum is zero
		//phiSum += phiSum*1.0005+0.001;
		currentT.lh = psiSum/sqrt(phiSum);
		currentT.flux = /*2.0*fluxPix**/ psiSum/phiSum;
		trajectory temp;
		for (int r=0; r<RESULTS_PER_PIXEL; ++r)
		{
			if ( currentT.lh > best[r].lh )
			{
				temp = best[r];
				best[r] = currentT;
				currentT = temp;
			}
		}
	}

	for (int r=0; r<RESULTS_PER_PIXEL; ++r)
	{
		results[ (y*width + x)*RESULTS_PER_PIXEL + r ] = best[r];
	}
}

__device__ float2 readPixel(float* img, int x, int y, int width, int height)
{
	float2 p; int i = y*width+x; p.x = img[i]; p.y = img[i+1];
	return p;
}

extern "C" void
deviceSearch(int trajCount, int imageCount, int psiPhiSize, int resultsCount,
			 trajectory * trajectoriesToSearch, trajectory *bestTrajects,
			 float *imageTimes, float *interleavedPsiPhi, long *dimensions)
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

	//dim3 blocks(dimensions[0],dimensions[1]);
	dim3 blocks(dimensions[0]/THREAD_DIM_X+1,dimensions[1]/THREAD_DIM_Y+1);
	dim3 threads(THREAD_DIM_X,THREAD_DIM_Y);

	// Launch Search
	searchImages<<<blocks, threads>>> (trajCount, dimensions[0],
		dimensions[1], imageCount, devicePsiPhi,
		deviceTests, deviceSearchResults, deviceImgTimes);

	// Read back results
	checkCudaErrors(cudaMemcpy(bestTrajects, deviceSearchResults,
				sizeof(trajectory)*resultsCount, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(deviceTests));
	checkCudaErrors(cudaFree(deviceImgTimes));
	checkCudaErrors(cudaFree(deviceSearchResults));
	checkCudaErrors(cudaFree(devicePsiPhi));
}


#endif /* KERNELS_CU_ */
