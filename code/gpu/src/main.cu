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
//#include <stdlib.h>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <math.h>
#include <dirent.h>
//#include <cstring>
//#include <vector>
#include <list>
//#include <memory>
//#include <algorithm>

#include <fitsio.h>
#include "GeneratorPSF.h"

#define THREAD_DIM_X 16
#define THREAD_DIM_Y 32
#define RESULTS_PER_PIXEL 8

using std::cout;

void readFitsImg(const char *name, long pixelsPerImage, float *target);

double readFitsMJD(const char *name);

void writeFitsImg(const char *name, long *dimensions, long pixelsPerImage, void *array);

void deviceConvolve(float *sourceImg, float *resultImg, long *dimensions, psfMatrix PSF);

std::string parseLine(std::ifstream& cFile, int debug);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/*
 * Represents a potential trajectory
 */
struct trajectory {
	// Trajectory velocities
	float xVel; 
	float yVel;
	// Likelyhood
	float lh;
	// Est. Flux
	float flux;
	// Origin
	int x; 
	int y;
	// Number of images summed
	//int itCount; 
};

/* 
 * For comparing trajectory structs, so that they can be sorted
 */
int compareTrajectory( const void * a, const void * b)
{
        return (int)(20000.0*(((trajectory*)b)->lh - ((trajectory*)a)->lh));
}

/*
 * Device kernel that convolves the provided image with the psf
 */
__global__ void convolvePSF(int width, int height, float *sourceImage, 
	float *resultImage, float *psf, int psfRad, int psfDim)
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
	float psfSum = 0.0;
	for (int j=minY; j<=maxY; ++j)
	{
		// #pragma unroll
		for (int i=minX; i<=maxX; ++i)
		{
			float currentPSF = psf[(j-minY)*psfDim+i-minX];
			psfSum += currentPSF;
			sum += (sourceImage[j*width+i]) * currentPSF;
		}
	}

	resultImage[y*width+x] = sum/psfSum;

}

/*
 * Searches through images (represented as a flat array of floats) looking for most likely
 * trajectories in the given list. Outputs a results image of best trajectories. Note that
 * for now only the single best trajectory starting at each pixel makes it to the results. 
 */
__global__ void searchImages(int trajectoryCount, int width, 
	int height, int imageCount, int edgePadding, float *psiPhiImages, 
	trajectory *trajectories, trajectory *results, float *imgTimes, 
	float slopeRejectThresh, float fluxPix, float termThreshold)
{

	// Get trajectory origin
	int x = blockIdx.x*THREAD_DIM_X+threadIdx.x;
	int y = blockIdx.y*THREAD_DIM_Y+threadIdx.y;
	
	trajectory best[RESULTS_PER_PIXEL];
	for (int r=0; r<RESULTS_PER_PIXEL; ++r)
	{
		best[r]  = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0, 
		.flux = 0.0, .x = x, .y = y/*, .itCount = 0*/ };
	}

	//if (x<width && y<height) results[ y*width + x ] = best;		

	// Give up if any trajectories will hit image edges
	if (x >= width || y >= height) 
	{
		return;
	}
	
	int pixelsPerImage = width*height;
	//int totalMemSize = width*height*imageCount*2;	
	
	// For each trajectory we'd like to search
	for (int t=0; t<trajectoryCount; ++t)
	{
	  	trajectory currentT = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0, 
		.flux = 0.0, .x = x, .y = y };
		/*float xVel = trajectories[t].xVel;
		float yVel = trajectories[t].yVel;
		float psiSum = 0.0;
		//float lastPsi = 10000.0;
		float phiSum = 0.0;
		*/
		currentT.xVel = trajectories[t].xVel;
		currentT.yVel = trajectories[t].yVel;
		float psiSum = 0.0;
		//float lastPsi = 10000.0;
		float phiSum = 0.0;

		// Sample each image at the appropriate pixel
		for (int i=0; i<imageCount; ++i)
		{
			int currentX = x + int(currentT.xVel*imgTimes[i]);
			int currentY = y + int(currentT.yVel*imgTimes[i]);
			if (currentX >= width || currentY >= height
			    || currentX < 0 || currentY < 0) 
			{	
				// Penalize trajctories that leave edge
				//psiSum += -0.1;
				continue;
			}
			int pixel = 2*(pixelsPerImage*i + 
				 currentY*width +
				 currentX);
	
			//float cPsi = min(psiPhiImages[pixel], 0.05);
			float cPsi = psiPhiImages[pixel];
			float cPhi = psiPhiImages[pixel+1];
			//float deltaPsi = cPsi-lastPsi;
			//if (deltaPsi<slopeRejectThresh)
			//{
				psiSum += min(cPsi,0.05);
			//	lastPsi = cPsi;
			//}
			phiSum += cPhi;
			
			//psiSum += min(psiPhiImages[pixel], 0.1);
			//phiSum += psiPhiImages[pixel+1];
			//best.itCount++;
			//if (psiSum <= 0.0 && i>4) break;
		}
		
		// Just in case a phiSum is zero
		//phiSum += phiSum*1.0005+0.001;
		currentT.lh = psiSum/sqrt(phiSum);
		currentT.flux = 2.0*fluxPix*psiSum/phiSum;
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
/*
__global__ void searchLocal(int trajectoryCount, int width, 
	int height, int imageCount, int edgePadding, float *psiPhiImages, 
	trajectory *trajectories, trajectory *results, float *imgTimes, 
	float slopeRejectThresh, float fluxPix)
{

	// Local memory to store nearby pixels 
	__shared__ float2 sA[64][64];
	__shared__ trajectory candidates[1024];
	
	// Center pixel coordinates
	int x = blockIdx.x;
	int y = blockIdx.y;
	// Thread indexes
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tIndex = 32*ty+tx;
	// Pixels needed from image
	int px = x+2*(tx-16);
	int py = y+2*(ty-16);
	
	// Pull needed pixels from global memory to shared
	
	if ( px >= 0 && py >= 0 && px+1 < width && py+1 < height)
	{
		sA[tx  ][ty  ] = readPixel(psiPhiImages, px,   py,   width, height);
		sA[tx+1][ty  ] = readPixel(psiPhiImages, px+1, py,   width, height);
		sA[tx  ][ty+1] = readPixel(psiPhiImages, px,   py+1, width, height);
		sA[tx+1][ty+1] = readPixel(psiPhiImages, px+1, py+1, width, height);
	}

	trajectory best = { .xVel = 0.0, .yVel = 0.0, .lh = 0.0, 
		.flux = 0.0, .x = x, .y = y, .itCount = trajectoryCount };

	int trajectsPerThread = (trajectoryCount/1024+1);
	int startIndex = trajectsPerThread*tIndex;
	int endIndex = trajectsPerThread*(tIndex+1);
	
	for (int t=startIndex; t<endIndex; t++)
	{
		
	}
	
	results[ y*width + x ] = best;
	
}
*/

int main(int argc, char* argv[])
{

	/* Read parameters from config file */
	std::ifstream pFile ("parameters.config");
    	if (!pFile.is_open()) 
		cout << "Unable to open parameters file." << '\n';

	long dimensions[2];
	using std::stoi;
	using std::stof;
	int debug             = stoi(parseLine(pFile, false));
	float psfSigma        = stof(parseLine(pFile, debug));
	float maskThreshold   = stof(parseLine(pFile, debug));
	float maskPenalty     = stof(parseLine(pFile, debug));
	int subtractAvg       = stoi(parseLine(pFile, debug));	
	float termThresh      = stof(parseLine(pFile, debug));
	float slopeReject     = stof(parseLine(pFile, debug));
	int angleSteps        = stoi(parseLine(pFile, debug));
	float minAngle        = stof(parseLine(pFile, debug));
	float maxAngle        = stof(parseLine(pFile, debug));
	int velocitySteps     = stoi(parseLine(pFile, debug));
	float minVelocity     = stof(parseLine(pFile, debug));
	float maxVelocity     = stof(parseLine(pFile, debug));
	int writeFiles        = stoi(parseLine(pFile, debug));	
	std::string realPath  = parseLine(pFile, debug);
	std::string psiPath   = parseLine(pFile, debug);
	std::string phiPath   = parseLine(pFile, debug);
	std::string rsltPath  = parseLine(pFile, debug);
	
	std::string params = "";
	std::string ln;
	pFile.seekg(0, pFile.beg);
	while(getline(pFile, ln))
	{
		params.append("# "+ln+"\n");
	}
	pFile.close();
     
	/* Create instances of psf and object generators */
	GeneratorPSF *gen = new GeneratorPSF();

	psfMatrix testPSF = gen->createGaussian(psfSigma);

	float psfCoverage = gen->printPSF(testPSF, debug);

	/* Read list of files from directory and get their dimensions  */
	std::list<std::string> fileNames;
	
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (realPath.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			std::string current = ent->d_name;
			if (current != "." && current != "..")
			{
				fileNames.push_back(realPath+current);
			//	printf ("%s\n", (realPath+current).c_str());
			}
		}	
	closedir (dir);
	}

	fileNames.sort();
	
	fitsfile *fptr1;
	int status = 0;
	int fileNotFound;
	// Read dimensions of image
	
	if (fits_open_file(&fptr1, (fileNames.front()+"[1]").c_str(),
		READONLY, &status)) fits_report_error(stderr, status);	
	if (fits_read_keys_lng(fptr1, "NAXIS", 1, 2, dimensions,
		&fileNotFound, &status)) fits_report_error(stderr, status);
	if (fits_close_file(fptr1, &status)) fits_report_error(stderr, status);
	
	int imageCount = fileNames.size();
	cout << "Reading " << imageCount << " images from " 
		<< realPath << "\n";

	
	/* Allocate pointers to images */
	long pixelsPerImage = dimensions[0] * dimensions[1];
	float **rawImages = new float*[imageCount];
	float **varianceImages = new float*[imageCount];
	float **maskImages = new float*[imageCount];

	float *imageTimes = new float[imageCount];

	// Load images from file
	double firstImageTime = readFitsMJD((fileNames.front()+"[0]").c_str());
	int imageIndex = 0;
	for (std::list<std::string>::iterator it=fileNames.begin();
		it != fileNames.end(); ++it)
	{
		// Allocate memory for each image
		rawImages[imageIndex] = new float[pixelsPerImage];
		varianceImages[imageIndex] = new float[pixelsPerImage];
		maskImages[imageIndex] = new float[pixelsPerImage];
		// Read Images
		readFitsImg((*it+"[1]").c_str(), pixelsPerImage, rawImages[imageIndex]);	
		readFitsImg((*it+"[2]").c_str(), pixelsPerImage, maskImages[imageIndex]);	
		readFitsImg((*it+"[3]").c_str(), pixelsPerImage, varianceImages[imageIndex]);			
		imageTimes[imageIndex] = (readFitsMJD((*it+"[0]").c_str())-firstImageTime);
		imageIndex++;
	}
	
	if (debug)
	{
		cout << "\nImage times: ";
		for (int i=0; i<imageCount; ++i)
		{	
			cout << imageTimes[i] << " ";
		}
		cout << "\n";	
	}
	
	if (debug) cout << "Masking images ... " << std::flush;
	// Create master mask
	float *masterMask = new float[pixelsPerImage];
	for (int i=0; i<imageCount; ++i)
	{
		for (int p=0; p<pixelsPerImage; ++p)
		{
			masterMask[p] += maskImages[i][p] == 0.0 ? 0.0 : 1.0;
		}
	}
	
	for (int p=0; p<pixelsPerImage; ++p)
	{
		masterMask[p] = masterMask[p]/float(imageCount) > maskThreshold ? 0.0 : 1.0;
	}

	// Mask Images. This part may be slow, could be moved to GPU ///
	
	#pragma omp parallel for 
	for (int i=0; i<imageCount; ++i)
	{
		// TODO: masks must be converted from ints to floats?
		// UPDATE: looks like this is done automatically by cfitsio
		
		// If maskThreshold is 0, use individual image masks rather than a master
		if (maskThreshold == 0.0) {
			for (int p=0; p<pixelsPerImage; ++p)
			{
				rawImages[i][p] = maskImages[i][p] == 0.0 ? 
					rawImages[i][p] / varianceImages[i][p] : maskPenalty;
			}
			for (int p=0; p<pixelsPerImage; ++p)
			{
				varianceImages[i][p] = maskImages[i][p] == 0.0 ? 
					1.0  / varianceImages[i][p] : 0.0;
			}
		} else {
			for (int p=0; p<pixelsPerImage; ++p)
			{
				rawImages[i][p] = masterMask[p] == 0.0 ? maskPenalty 
					: rawImages[i][p] / varianceImages[i][p];
			}
			for (int p=0; p<pixelsPerImage; ++p)
			{
				varianceImages[i][p] = masterMask[p] / varianceImages[i][p];
			}
		}
	}

	if (debug) cout << "Done.\n";

	// Free mask images memory
	for (int i=0; i<imageCount; ++i)
	{
		delete[] maskImages[i];
	}
	delete[] maskImages;
	delete[] masterMask;
	
	float **psiImages = new float*[imageCount];
	float **phiImages = new float*[imageCount];

	/* Generate psi and phi images on device */
	
	if (debug) cout << "Creating Psi and Phi ... " << std::flush;
	std::clock_t t1 = std::clock();

	for (int i=0; i<imageCount; ++i)
	{
		psiImages[i] = new float[pixelsPerImage];
		phiImages[i] = new float[pixelsPerImage];
		deviceConvolve(rawImages[i], psiImages[i], dimensions, testPSF);	
		deviceConvolve(varianceImages[i], phiImages[i], dimensions, testPSF);
		deviceConvolve(phiImages[i], phiImages[i], dimensions, testPSF);
	}

	std::clock_t t2 = std::clock();

	if (debug) cout << "Done. Took " << 1000.0*(t2 - t1)/(double) 
		(CLOCKS_PER_SEC*imageCount) << " ms per image\n";
	
	
	// Subtract average (very simple difference imaging)
	if (subtractAvg)
	{
		float *avgPsi = new float[pixelsPerImage];	
		for (int i=0; i<imageCount; i++)
		{
			for (int p=0; p<pixelsPerImage; p++)
			{
				avgPsi[p] += psiImages[i][p];
			} 
		}
	
		for (int i=0; i<imageCount; i++)
		{
			for (int p=0; p<pixelsPerImage; p++)
			{
				psiImages[i][p] -= avgPsi[p] / float(imageCount); 
			}
		}
		delete[] avgPsi;
	}

	// Write images to file 
	if (writeFiles)
	{
		cout << "Writing images to file... " << std::flush;
		std::stringstream ss;
		for (int writeIndex=0; writeIndex<imageCount; ++writeIndex)
		{
			/* Create file name */
			ss << psiPath << "T";
			// Add leading zeros to filename
			if (writeIndex+1<100) ss << "0";
			if (writeIndex+1<10) ss << "0";
			ss << writeIndex+1 << "psi.fits";
			writeFitsImg(ss.str().c_str(), dimensions, 
				pixelsPerImage, psiImages[writeIndex]);
			ss.str("");
			ss.clear();		

			ss << phiPath << "T";
			if (writeIndex+1<100) ss << "0";
			if (writeIndex+1<10) ss << "0"; 
			ss << writeIndex+1 << "phi.fits";
			writeFitsImg(ss.str().c_str(), dimensions, 
				pixelsPerImage, phiImages[writeIndex]);
			ss.str("");
			ss.clear();
		}
	}
	cout << "Done.\n";

	
	if (debug) cout << "Creating interleaved psi/phi buffer ... ";
	// Create interleaved psi/phi image buffer for fast lookup on GPU
	// Hopefully we have enough RAM for this..
	int psiPhiSize = 2*pixelsPerImage*imageCount;
	float *interleavedPsiPhi = new float[psiPhiSize];
	#pragma omp parallel for
	for (int i=0; i<imageCount; ++i)
	{
		for (int p=0; p<pixelsPerImage; ++p)
		{
			int pixel = i*pixelsPerImage+p;
			interleavedPsiPhi[2*pixel + 0] = psiImages[i][p];
			interleavedPsiPhi[2*pixel + 1] = phiImages[i][p];
		}
	}
	if (debug) cout << "Done.\n";	
	
	/* Free raw images a psi/phi images */
	for (int im=0; im<imageCount; ++im)
	{
		delete[] rawImages[im];
		delete[] varianceImages[im];
		delete[] psiImages[im];
		delete[] phiImages[im];
	}

	delete[] rawImages;
	delete[] varianceImages;
	delete[] psiImages;
	delete[] phiImages;


	///* Search images on GPU *///
	
		
	/* Create trajectories to search */
	float *angles = new float[angleSteps];
	float da = (maxAngle-minAngle)/float(angleSteps);
	for (int i=0; i<angleSteps; ++i)
	{
		angles[i] = minAngle+float(i)*da;
	}

	float *velocities = new float[velocitySteps];
	float dv = (maxVelocity-minVelocity)/float(velocitySteps);
	for (int i=0; i<velocitySteps; ++i)
	{
		velocities[i] = minVelocity+float(i)*dv;	
	}	
 
	int trajCount = angleSteps*velocitySteps;
	trajectory *trajectoriesToSearch = new trajectory[trajCount];
	for (int a=0; a<angleSteps; ++a)
	{
		for (int v=0; v<velocitySteps; ++v)
		{
			trajectoriesToSearch[a*velocitySteps+v].xVel = cos(angles[a])*velocities[v];
			trajectoriesToSearch[a*velocitySteps+v].yVel = sin(angles[a])*velocities[v]; 
		}
	}
	
	/* Prepare Search */
	
	std::clock_t t3 = std::clock();

	// assumes object is not moving more than 2 pixels per image
	int padding = 2*imageCount+int(psfSigma)+1;
	cout << "Searching " << trajCount << " possible trajectories starting from " 
		<< ((dimensions[0])*(dimensions[1])) << " pixels... " << "\n";

	// Allocate Host memory to store results in
	int resultsCount = pixelsPerImage*RESULTS_PER_PIXEL;
	trajectory* bestTrajects = new trajectory[resultsCount];

	// Allocate Device memory 
	trajectory *deviceTests;
	float *deviceImgTimes;
	float *devicePsiPhi;
	trajectory *deviceSearchResults;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceTests, sizeof(trajectory)*trajCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceImgTimes, sizeof(float)*imageCount));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&devicePsiPhi, 
		sizeof(float)*psiPhiSize));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceSearchResults, 
		sizeof(trajectory)*resultsCount));
	

	// Copy trajectories to search
	CUDA_CHECK_RETURN(cudaMemcpy(deviceTests, trajectoriesToSearch,
			sizeof(trajectory)*trajCount, cudaMemcpyHostToDevice));


	CUDA_CHECK_RETURN(cudaMemcpy(deviceImgTimes, imageTimes,
			sizeof(float)*imageCount, cudaMemcpyHostToDevice));

	// Copy over interleaved buffer of psi and phi images
	CUDA_CHECK_RETURN(cudaMemcpy(devicePsiPhi, interleavedPsiPhi,
		sizeof(float)*psiPhiSize, cudaMemcpyHostToDevice));

	//dim3 blocks(dimensions[0],dimensions[1]);
	dim3 blocks(dimensions[0]/THREAD_DIM_X+1,dimensions[1]/THREAD_DIM_Y+1);
	dim3 threads(THREAD_DIM_X,THREAD_DIM_Y);
	
	int halfPSF = testPSF.dim/2;
	float fluxPix = 1.0 / testPSF.kernel[halfPSF*testPSF.dim+halfPSF]; 

	// Launch Search
	searchImages<<<blocks, threads>>> (trajCount, dimensions[0], 
		dimensions[1], imageCount, padding, devicePsiPhi,
		deviceTests, deviceSearchResults, deviceImgTimes, 
		slopeReject, fluxPix, termThresh);

	// Read back results
	CUDA_CHECK_RETURN(cudaMemcpy(bestTrajects, deviceSearchResults,
				sizeof(trajectory)*resultsCount, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(deviceTests));
	CUDA_CHECK_RETURN(cudaFree(deviceImgTimes));
	CUDA_CHECK_RETURN(cudaFree(deviceSearchResults));
	CUDA_CHECK_RETURN(cudaFree(devicePsiPhi));


	std::clock_t t4 = std::clock();

	cout << "Took " << 1.0*(t4 - t3)/(double) (CLOCKS_PER_SEC)
		  << " seconds to complete search.\n"; 

	// Sort results by likelihood
	cout << "Sorting results... " << std::flush;
	qsort(bestTrajects, resultsCount, sizeof(trajectory), compareTrajectory);
	cout << "Done.\n" << std::flush;	

	if (debug)
	{
		for (int i=0; i<15; ++i)
		{
			if (i+1 < 10) cout << " ";
			cout << i+1 << ". Likelihood: "  << bestTrajects[i].lh 
				  << " at x: " << bestTrajects[i].x << ", y: " << bestTrajects[i].y
				  << "  and velocity x: " << bestTrajects[i].xVel 
				  << ", y: " << bestTrajects[i].yVel << " Est. Flux: "
				  << bestTrajects[i].flux <<"\n" ;
		}
	}

	/* Write results to file */
	// cout needs to be rerouted to output to console after this...
	
	//int namePos = realPath.find_last_of("/")-1;
	//std::string resultFile = realPath.substr(namePos,namePos);

	std::freopen(rsltPath.c_str(), "w", stdout);
	cout << "# t0_x t0_y theta_par theta_perp v_x v_y likelihood est_flux\n";
	cout << params;
        for (int i=0; i<resultsCount / 12  /* / 8 */; ++i)
        {
                cout << bestTrajects[i].x << " " << bestTrajects[i].y << " 0.0 0.0 "
                          << bestTrajects[i].xVel << " " << bestTrajects[i].yVel << " "       
                          << bestTrajects[i].lh << " "  <<  bestTrajects[i].flux << "\n" ;
        }

	// Finished!

	/* Free remaining memory */
	delete[] imageTimes;
	delete[] interleavedPsiPhi;		

	delete[] angles;
	delete[] velocities;
	delete[] trajectoriesToSearch;	
	delete[] bestTrajects;
	
	delete gen;

	return 0;
} 

std::string parseLine(std::ifstream& pFile, int debug)
{
	std::string line;
	getline(pFile, line);
        // Read entry to the right of the ":" symbol
	int delimiterPos = line.find(":");
	if (debug) 
	{
		cout << line.substr(0, delimiterPos );
		cout << " : " << line.substr(delimiterPos + 2) << "\n";
	}
	return line.substr(delimiterPos + 2);
}

void readFitsImg(const char *name, long pixelsPerImage, float *target)
{
	fitsfile *fptr;
	int nullval = 0;
	int anynull;
	int status = 0;
	
	if (fits_open_file(&fptr, name, READONLY, &status)) ffrprt(stderr, status);
        if (fits_read_img(fptr, TFLOAT, 1, pixelsPerImage, 
		&nullval, target, &anynull, &status)) ffrprt(stderr, status);
        if (fits_close_file(fptr, &status)) ffrprt(stderr, status);

	
}

double readFitsMJD(const char *name)
{
	int status = 0;
        fitsfile *fptr;
	double time;
	if (fits_open_file(&fptr, name, READONLY, 
		&status)) ffrprt(stderr, status);
        if (fits_read_key(fptr, TDOUBLE, "MJD", &time, 
		NULL, &status)) ffrprt(stderr, status);
        if (fits_close_file(fptr, &status)) ffrprt(stderr, status);
	return time;
}

void writeFitsImg(const char *name, long *dimensions, long pixelsPerImage, void *array)
{
	int status = 0;
	fitsfile *f;
        /* Create file with name */
	fits_create_file(&f, name, &status);

	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, array, &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void deviceConvolve(float *sourceImg, float *resultImg, long *dimensions, psfMatrix PSF)
{
	// Pointers to device memory //
	float *deviceKernel;
	float *deviceSourceImg;
	float *deviceResultImg;

	long pixelsPerImage = dimensions[0]*dimensions[1];
	dim3 blocks(dimensions[0]/32+1,dimensions[1]/32+1);
	dim3 threads(32,32);

	// Allocate Device memory
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceKernel, sizeof(float)*PSF.dim*PSF.dim));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceSourceImg, sizeof(float)*pixelsPerImage));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceResultImg, sizeof(float)*pixelsPerImage));

	CUDA_CHECK_RETURN(cudaMemcpy(deviceKernel, PSF.kernel,
		sizeof(float)*PSF.dim*PSF.dim, cudaMemcpyHostToDevice));

	
	CUDA_CHECK_RETURN(cudaMemcpy(deviceSourceImg, sourceImg,
		sizeof(float)*pixelsPerImage, cudaMemcpyHostToDevice));

	convolvePSF<<<blocks, threads>>> (dimensions[0], dimensions[1], 
		deviceSourceImg, deviceResultImg, deviceKernel, PSF.dim/2, PSF.dim);

	CUDA_CHECK_RETURN(cudaMemcpy(resultImg, deviceResultImg,
		sizeof(float)*pixelsPerImage, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(deviceKernel));
	CUDA_CHECK_RETURN(cudaFree(deviceSourceImg));
	CUDA_CHECK_RETURN(cudaFree(deviceResultImg));

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

