/*
 ============================================================================
 Name        : KBMOD CUDA
 Author      : Peter Whidden
 Description :
 ============================================================================
 */

#include <iostream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <math.h>
#include <dirent.h>
//#include <cstring>
//#include <vector>
//#include <memory>
//#include <algorithm>

#include <fitsio.h>
#include "GeneratorPSF.h"
#include "FakeAsteroid.h"

void writeFitsImg(const char *name, 
	long *dimensions, long pixelsPerImage, void *array);

const char* parseLine(std::ifstream& cFile, int debug);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/*
 * A struct to represent a potential trajectory
 */
struct trajectory {
	// Trajectory velocities
	float xVel; 
	float yVel;
	// Likelyhood
	float lh;
	// Origin
	int x; 
	int y;
	// Number of images summed
	int itCount; 
};

/* 
 * For comparing trajectory structs, so that they can be sorted
 */
int compareTrajectory( const void * a, const void * b)
{
        return (int)(5000.0*(((trajectory*)b)->lh - ((trajectory*)a)->lh));
}

/*
 * Device kernel that compares the provided PSF distribution to the distribution
 * around each pixel in the provided image
 */
__global__ void convolvePSF(int width, int height, int imageCount,
	float *image, float *psiImagess, float *psf, int psfRad, 
	int psfDim, float background, float normalization)
{
	// Find bounds of convolution area
	const int x = blockIdx.x*32+threadIdx.x;
	const int y = blockIdx.y*32+threadIdx.y;
	if (x < 0 || x > width || y < 0 || y > height) return;
	const int minX = max(x-psfRad, 0);
	const int minY = max(y-psfRad, 0);
	const int maxX = min(x+psfRad, width-1);
	const int maxY = min(y+psfRad, height-1);
	const int dx = maxX-minX;
	const int dy = maxY-minY;
	if (dx < psfRad || dy < psfRad ) return;
 
	// Read kernel
	float sumDifference = 0.0;
	for (int j=minY; j<=maxY; ++j)
	{
		// #pragma unroll
		for (int i=minX; i<=maxX; ++i)
		{
			sumDifference += (image[j*width+i] - background)
					 * psf[(j-minY)*psfDim+i-minX];
		}
	}

	psiImagess[y*width+x] = sumDifference*normalization;

}

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a psiImagess image of best trajectories. Note that
 * for now only the single best trajectory starting at each pixel makes it to psiImagess. 
 */
__global__ void searchImages(int width, int height, int imageCount, float *images, 
	int trajectoryCount, trajectory *tests, trajectory *psiImagess, float mean, int edgePadding)
{

	// Get trajectory origin
	int x = blockIdx.x*32+threadIdx.x;
	int y = blockIdx.y*32+threadIdx.y;
	// Give up if any trajectories will hit image edges
	if (x < edgePadding || x + edgePadding > width ||
	    y < edgePadding || y + edgePadding > height) return;

	trajectory best = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0, 
		.x = x, .y = y, .itCount = trajectoryCount };
	
	for (int t=0; t<trajectoryCount; ++t)
	{
		float xVel = tests[t].xVel;
		float yVel = tests[t].yVel;
		float currentLikelyhood = 0.0;
		for (int i=0; i<imageCount; ++i)
		{
			currentLikelyhood += images[ i*width*height + 
				(y+int( yVel*float(i)))*width +
				 x+int( xVel*float(i)) ] / mean; 	
		}
		
		if ( currentLikelyhood > best.lh )
		{
			best.lh = currentLikelyhood;
			best.xVel = xVel;
			best.yVel = yVel;
		}		
	}	
	
	psiImagess[ y*width + x ] = best;	
}


int main(int argc, char* argv[])
{

	/* Read parameters from config file */
	std::ifstream pFile ("parameters.config");
    	if (!pFile.is_open()) 
		std::cout << "Unable to open parameters file." << '\n';
	
	int debug             = atoi(parseLine(pFile, false));
	int imageCount        = atoi(parseLine(pFile, debug));
	int generateImages    = atoi(parseLine(pFile, debug));
	int imgWidth          = atoi(parseLine(pFile, debug));
	int imgHeight         = atoi(parseLine(pFile, debug));
	float psfSigma        = atof(parseLine(pFile, debug));
	float asteroidLevel   = atof(parseLine(pFile, debug));
	float initialX        = atof(parseLine(pFile, debug));
	float initialY        = atof(parseLine(pFile, debug));
	float velocityX       = atof(parseLine(pFile, debug));
	float velocityY       = atof(parseLine(pFile, debug));
	float backgroundLevel = atof(parseLine(pFile, debug));
	float backgroundSigma = atof(parseLine(pFile, debug));
	int anglesCount       = atoi(parseLine(pFile, debug));
	int velocitySteps     = atoi(parseLine(pFile, debug));
	float minVelocity     = atof(parseLine(pFile, debug));
	float maxVelocity     = atof(parseLine(pFile, debug));
	int writeFiles        = atoi(parseLine(pFile, debug));
	std::string realPath  = parseLine(pFile, debug);
	std::string origPath  = parseLine(pFile, debug);
	std::string psiPath   = parseLine(pFile, debug);
	pFile.close();
     
	/* Create instances of psf and object generators */
	GeneratorPSF *gen = new GeneratorPSF();

	psfMatrix testPSF = gen->createGaussian(psfSigma);

	float psfCoverage = gen->printPSF(testPSF, debug);

	FakeAsteroid *asteroid = new FakeAsteroid();

	/* Setup Image/FITS Properties of test Images  */
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (realPath.c_str())) != NULL) {
  		/* print all the files and directories within directory */
 		while ((ent = readdir (dir)) != NULL) {
    			
			printf ("%s\n", ent->d_name);
  		}	
  		closedir (dir);
	}
	

	long pixelsPerImage;
	
	long dimensions[2] = { imgWidth, imgHeight }; // X and Y dimensions
	pixelsPerImage = dimensions[0] * dimensions[1];
	std::stringstream ss;
	float **pixelArray = new float*[imageCount];

	// Create asteroid images //
	for (int imageIndex=0; imageIndex<imageCount; ++imageIndex)
	{
		/* Initialize the values in the image with noisy astro */
		pixelArray[imageIndex] = new float[pixelsPerImage];
		asteroid->createImage( pixelArray[imageIndex], dimensions[0], dimensions[1],
	 	    	velocityX*float(imageIndex)+initialX,  // Asteroid X position 
			velocityY*float(imageIndex)+initialY,  // Asteroid Y position
			testPSF, asteroidLevel, backgroundLevel, backgroundSigma);
	}

	/*
	 *  TODO: Loading real FITS images would go here
	 */

	/* Generate psi images on device */

	std::clock_t t1 = std::clock();

	// Pointers to device memory //
	float **psiImages = new float*[pixelsPerImage];
	float *devicePsf;
	float *deviceOriginalImage;
	float *devicePsiImage;

	dim3 blocks(dimensions[0]/32+1,dimensions[1]/32+1);
	dim3 threads(32,32);

	// Allocate Device memory
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devicePsf, sizeof(float)*testPSF.dim*testPSF.dim));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceOriginalImage, sizeof(float)*pixelsPerImage));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devicePsiImage, sizeof(float)*pixelsPerImage));

	CUDA_CHECK_RETURN(cudaMemcpy(devicePsf, testPSF.kernel,
		sizeof(float)*testPSF.dim*testPSF.dim, cudaMemcpyHostToDevice));

	for (int procIndex=0; procIndex<imageCount; ++procIndex)
	{
		psiImages[procIndex] = new float[pixelsPerImage];
		// Copy image to
		CUDA_CHECK_RETURN(cudaMemcpy(deviceOriginalImage, pixelArray[procIndex],
			sizeof(float)*pixelsPerImage, cudaMemcpyHostToDevice));

		convolvePSF<<<blocks, threads>>> (dimensions[0], dimensions[1], 
			imageCount, deviceOriginalImage, devicePsiImage, devicePsf, 
			testPSF.dim/2, testPSF.dim, backgroundLevel, 1.0/psfCoverage);

		CUDA_CHECK_RETURN(cudaMemcpy(psiImages[procIndex], devicePsiImage,
			sizeof(float)*pixelsPerImage, cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK_RETURN(cudaFree(devicePsf));
	CUDA_CHECK_RETURN(cudaFree(deviceOriginalImage));
	CUDA_CHECK_RETURN(cudaFree(devicePsiImage));

	std::clock_t t2 = std::clock();

	std::cout << imageCount << " images, convolution took " <<
		1000.0*(t2 - t1)/(double) (CLOCKS_PER_SEC*imageCount) 
		  << " ms per image\n";


	///* Search images on GPU *///
	
	std::clock_t t3 = std::clock();
		
	/* Create test trajectories */
	float *angles = new float[anglesCount];
	for (int i=0; i<anglesCount; ++i)
	{
		angles[i] = 6.283185*float(i)/float(anglesCount);
	}

	float *velocities = new float[velocitySteps];
	float dv = (maxVelocity-minVelocity)/float(velocitySteps);
	for (int i=0; i<velocitySteps; ++i)
	{
		velocities[i] = minVelocity+float(i)*dv;	
	}	
 
	int trajCount = anglesCount*velocitySteps;
	trajectory *trajectoriesToSearch = new trajectory[trajCount];
	for (int a=0; a<anglesCount; ++a)
	{
		for (int v=0; v<velocitySteps; ++v)
		{
			trajectoriesToSearch[a*velocitySteps+v].xVel = cos(angles[a])*velocities[v];
			trajectoriesToSearch[a*velocitySteps+v].yVel = sin(angles[a])*velocities[v]; 
		}
	}
	
	/* Prepare Search */
	
	// assumes object is not moving more than 2 pixels per image
	int padding = 2*imageCount+int(psfSigma)+1;
	std::cout << "Searching " << trajCount << " possible trajectories starting from " 
		<< ((dimensions[0]-padding)*(dimensions[1]-padding)) << " pixels... " << "\n";

	// Allocate Host memory to store psiImagess in
	trajectory* trajResult = new trajectory[pixelsPerImage];

	// Allocate Device memory 
	trajectory *deviceTests;
	trajectory *deviceSearchResults;
	float *deviceImages;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceTests, sizeof(trajectory)*trajCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImages, sizeof(float)*pixelsPerImage*imageCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceSearchResults, sizeof(trajectory)*pixelsPerImage));
	

	// Copy trajectories to search
	CUDA_CHECK_RETURN(cudaMemcpy(deviceTests, trajectoriesToSearch,
			sizeof(trajectory)*trajCount, cudaMemcpyHostToDevice));

	// Copy over psi images one at a time
	for (int i=0; i<imageCount; ++i)
	{
		CUDA_CHECK_RETURN(cudaMemcpy(deviceImages+pixelsPerImage*i, psiImages[i],
			sizeof(float)*pixelsPerImage, cudaMemcpyHostToDevice));
	}

	// Launch Search
	searchImages<<<blocks, threads>>> (dimensions[0], dimensions[1], imageCount, deviceImages,
				trajCount, deviceTests, deviceSearchResults, backgroundLevel, padding);

	// Read back psiImagess
	CUDA_CHECK_RETURN(cudaMemcpy(trajResult, deviceSearchResults,
				sizeof(trajectory)*pixelsPerImage, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(deviceTests));
	CUDA_CHECK_RETURN(cudaFree(deviceSearchResults));
	CUDA_CHECK_RETURN(cudaFree(deviceImages));

	
	// Sort psiImagess by likelihood
	qsort(trajResult, pixelsPerImage, sizeof(trajectory), compareTrajectory);
	if (debug)
	{
		for (int i=0; i<15; ++i)
		{
			if (i+1 < 10) std::cout << " ";
			std::cout << i+1 << ". Likelihood: "  << trajResult[i].lh 
				  << " at x: " << trajResult[i].x << ", y: " << trajResult[i].y
				  << "  and velocity x: " << trajResult[i].xVel 
				  << ", y: " << trajResult[i].yVel << "\n" ;
		}
	}

	std::clock_t t4 = std::clock();

	std::cout << "Took " << 1.0*(t4 - t3)/(double) (CLOCKS_PER_SEC)
		  << " seconds to complete search.\n"; 
	std::cout << "Writing images to file... ";

	// Write images to file 
	if (writeFiles)
	{
		for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
		{
			/* Create file name */
			ss << origPath << "T";
			// Add leading zeros to filename
			if (writeIndex+1<100) ss << "0";
			if (writeIndex+1<10) ss << "0";
			ss << writeIndex+1 << ".fits";
			writeFitsImg(ss.str().c_str(), dimensions, 
				pixelsPerImage, pixelArray[writeIndex]);
			ss.str("");
			ss.clear();		

			ss << psiPath << "T";
			if (writeIndex+1<100) ss << "0";
			if (writeIndex+1<10) ss << "0"; 
			ss << writeIndex+1 << "psi.fits";
			writeFitsImg(ss.str().c_str(), dimensions, 
				pixelsPerImage, psiImages[writeIndex]);
			ss.str("");
			ss.clear();
		}
	}
	std::cout << "Done.\n";

	/* Write psiImagess file */
	// std::cout needs to be rerouted to output to console after this...
	std::freopen("results.txt", "w", stdout);
	std::cout << "# t0_x t0_y theta_par theta_perp v_x v_y likelihood est_flux\n";
        for (int i=0; i<20; ++i)
        {
                std::cout << trajResult[i].x << " " << trajResult[i].y << " 0.0 0.0 "
                          << trajResult[i].xVel << " " << trajResult[i].yVel << " "       
                          << trajResult[i].lh << " 0.0\n" ;
        }

	// Finished!

	/* Free memory */
	for (int im=0; im<imageCount; ++im)
	{
		delete[] pixelArray[im];
		delete[] psiImages[im];
	}

	delete[] pixelArray;
	delete[] psiImages;
	
	delete[] angles;
	delete[] velocities;
	delete[] trajectoriesToSearch;	
	delete[] trajResult;
	
	delete gen;
	delete asteroid;

	return 0;
} 

const char* parseLine(std::ifstream& pFile, int debug)
{
	std::string line;
	getline(pFile, line);
        // Read entry to the right of the ":" symbol
	int delimiterPos = line.find(":");
	if (debug) 
	{
		std::cout << line.substr(0, delimiterPos );
		std::cout << " : " << line.substr(delimiterPos + 2) << "\n";
	}
	return (line.substr(delimiterPos + 2)).c_str();
}

void writeFitsImg(const char *name, long *dimensions, long pixelsPerImage, void *array)
{
	/* initialize status before calling fitsio routines */
	int status = 0;
	fitsfile *f;
        /* Create file with name */
	fits_create_file(&f, name, &status);

	/* Create the primary array image (32-bit float pixels */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, array, &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err)
			<< "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

