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

#include <ctime>
#include <math.h>

//#include <cstring>
#include <vector>
#include <list>
//#include <memory>
#include <parallel/algorithm>

#include "fitsutil.h"
#include "common.h"
#include "ImageStack.h"

using std::cout;

extern "C" void
deviceConvolve(float *sourceImg, float *resultImg,
			   long *dimensions, PointSpreadFunc PSF, float maskFlag);

extern "C" void
deviceSearch(int trajCount, int imageCount, int psiPhiSize, int resultsCount,
			 trajectory * trajectoriesToSearch, trajectory *bestTrajects,
		     float *imageTimes, float *interleavedPsiPhi, long *dimensions);


std::string parseLine(std::ifstream& cFile, int debug);


int main(int argc, char* argv[])
{

	std::clock_t setupA = std::clock();

	/* Read parameters from config file */
	std::ifstream pFile (argv[1]);
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

	PointSpreadFunc psf(psfSigma);
	PointSpreadFunc psfSQ(psfSigma);

	psf.printPSF(debug);

	ImageStack imStack(realPath, debug);

	imStack.loadImages();
	imStack.applyMasterMask(0, 4);
	imStack.applyMaskFlags(0);

	//RawImage img("file.fits");
	//img.applyMaskFlags(1);

	/* Allocate pointers to images */
	/*
	long pixelsPerImage = dimensions[0] * dimensions[1];
	float **rawImages = new float*[imageCount];
	float **varianceImages = new float*[imageCount];
	float **maskImages = new float*[imageCount];

	float *imageTimes = new float[imageCount];
	*/

	// Allocate memory for cfitsio

	/*

	if (debug) cout << "Masking images ... " << std::flush;
	// Create master mask
	float *masterMask = new float[pixelsPerImage];
	for (int i=0; i<imageCount; ++i)
	{
		for (int p=0; p<pixelsPerImage; ++p)
		{
			masterMask[p] += ((int(maskImages[i][p]) &
				0x00000020) == 0x00000020) ? 1.0 : 0.0;
		}
	}

	for (int i=0; i<imageCount; ++i)
	{
		// TODO: masks must be converted from ints to floats?
		// UPDATE: looks like this is done automatically by cfitsio

		// Use individual masks for anything other than 0 or 32
 		// (0x00000000 or 0x00000020)
		// Master mask for any pixels that are masked
		// more than maskThreshold times
		for (int p=0; p<pixelsPerImage; ++p)
		{
			int cur = int(maskImages[i][p]);
			if ((cur == 0 || cur == 32 || cur == 39 || cur == 37)
				&& masterMask[p] < maskThreshold) {
				rawImages[i][p] =
					rawImages[i][p] / varianceImages[i][p];
				varianceImages[i][p] = 1.0 / varianceImages[i][p];
			} else {
				rawImages[i][p] = MASK_FLAG;
				varianceImages[i][p] = MASK_FLAG;
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

	/* Generate psi and phi images on device * /

	if (debug) cout << "Creating Psi and Phi ... " << std::flush;
	std::clock_t t1 = std::clock();

	for (int i=0; i<imageCount; ++i)
	{
		psiImages[i] = new float[pixelsPerImage];
		phiImages[i] = new float[pixelsPerImage];
		deviceConvolve(rawImages[i], psiImages[i],
			dimensions, testPSF, maskPenalty);
		deviceConvolve(varianceImages[i], phiImages[i],
			dimensions, testPSFSQ, maskPenalty);
	}

	std::clock_t t2 = std::clock();

	if (debug) cout << "Done. Took " << 1000.0*(t2 - t1)/(double)
		(CLOCKS_PER_SEC*imageCount) << " ms per image\n";


	// Set pixels flagged as mask to 0.0
	for (int i=0; i<imageCount; ++i)
	{
		for (int p=0; p<pixelsPerImage; ++p)
		{
			psiImages[i][p] = psiImages[i][p] < MASK_FLAG/2 ?
				0.0 : psiImages[i][p];
			phiImages[i][p] = phiImages[i][p] < MASK_FLAG/2 ?
				0.0 : phiImages[i][p];
		}
	}

	// Write images to file
	if (writeFiles)
	{
		cout << "Writing images to file... " << std::flush;
		writeImageBatch(imageCount, psiPath, phiPath,
						pixelsPerImage, psiImages, phiImages, dimensions);
	}
	cout << "Done.\n";


	if (debug) cout << "Creating interleaved psi/phi buffer ... ";
	// Create interleaved psi/phi image buffer for fast lookup on GPU
	// Hopefully we have enough RAM for this...
	int psiPhiSize = 2*pixelsPerImage*imageCount;
	float *interleavedPsiPhi = new float[psiPhiSize];

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

	/* Free raw images a psi/phi images * /
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


	///* Search images on GPU * /

	/* Create trajectories to search * /
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

	std::clock_t setupB = std::clock();
	cout << "Setup took a total of " <<
			(setupB-setupA)/(double)(CLOCKS_PER_SEC) << " seconds.\n";

	/* Prepare Search * /

	std::clock_t t3 = std::clock();

	cout << "Searching " << trajCount << " possible trajectories starting from "
		<< ((dimensions[0])*(dimensions[1])) << " pixels... " << "\n";

	// Allocate Host memory to store results in
	int resultsCount = pixelsPerImage*RESULTS_PER_PIXEL;
	trajectory* bestTrajects = new trajectory[resultsCount];

	deviceSearch(trajCount, imageCount, psiPhiSize, resultsCount,
				trajectoriesToSearch, bestTrajects,
				imageTimes, interleavedPsiPhi, dimensions);

	std::clock_t t4 = std::clock();

	cout << "Took " << 1.0*(t4 - t3)/(double) (CLOCKS_PER_SEC)
		  << " seconds to complete search.\n";

	// Sort results by likelihood
	cout << "Sorting results... " << std::flush;
	std::clock_t t5 = std::clock();
	std::vector<trajectory> bestResults (bestTrajects, bestTrajects+resultsCount);
	__gnu_parallel::sort(bestResults.begin(), bestResults.end(),
			[](trajectory a, trajectory b) {
		return b.lh < a.lh;
	});
	std::clock_t t6 = std::clock();
	cout << "Done. Took " << (t6-t5)/(double)(CLOCKS_PER_SEC)
			<< " core seconds to sort.\n" << std::flush;

	if (debug)
	{
		for (int i=0; i<5; ++i)
		{
			if (i+1 < 10) cout << " ";
			cout << i+1 << ". Likelihood: "  << bestResults[i].lh
				  << " at x: " << bestResults[i].x << ", y: " << bestResults[i].y
				  << "  and velocity x: " << bestResults[i].xVel
				  << ", y: " << bestResults[i].yVel << " Est. Flux: "
				  << bestResults[i].flux <<"\n" ;
		}
	}

	/* Write results to file * /
	// cout needs to be rerouted to output to console after this...

	//int namePos = realPath.find_last_of("/")-1;
	//std::string resultFile = realPath.substr(namePos,namePos);

	std::freopen(rsltPath.c_str(), "w", stdout);
	cout << "# t0_x t0_y theta_par theta_perp v_x v_y likelihood est_flux\n";
	cout << params;
        for (int i=0; i<resultsCount / 12  /* / 8 * /; ++i)
        {
                cout << bestResults[i].x << " " << bestResults[i].y << " 0.0 0.0 "
                     << bestResults[i].xVel << " " << bestResults[i].yVel << " "
                     << bestResults[i].lh << " "  <<  bestResults[i].flux << "\n" ;
        }

	// Finished!

	/* Free remaining memory * /
	delete[] imageTimes;
	delete[] interleavedPsiPhi;

	delete[] angles;
	delete[] velocities;
	delete[] trajectoriesToSearch;
	delete[] bestTrajects;
	
	delete gen;

	return 0;

	*/
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





