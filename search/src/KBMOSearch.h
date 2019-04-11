/*
 * KBMOSearch.h
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#ifndef KBMODSEARCH_H_
#define KBMODSEARCH_H_

#include <parallel/algorithm>
#include <algorithm>
#include <functional>
#include <queue>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
//#include <stdio.h>
#include <assert.h>
#include <float.h>
#include "common.h"
#include "PointSpreadFunc.h"
#include "ImageStack.h"

namespace kbmod {

extern "C" void
deviceSearch(int trajCount, int imageCount, int minObservations, int psiPhiSize,
			 int resultsCount, trajectory *trajectoriesToSearch, trajectory *bestTrajects,
		     float *imageTimes, float *interleavedPsiPhi, int width, int height);

extern "C" void
devicePooledSetup(int imageCount, int depth, float *times, int *dimensions, float *interleavedImages,
		float **deviceTimes, float **deviceImages, int **deviceDimensions); // Dimensions?

extern "C" void
devicePooledTeardown(float **deviceTimes, float **deviceImages, int **dimensions);

extern "C" void
deviceLHBatch(int imageCount, int depth, int regionCount, trajRegion *regions,
		float **deviceTimes, float **deviceImages, float **deviceDimensions);

class KBMOSearch {
public:
	KBMOSearch(ImageStack& imstack, PointSpreadFunc& PSF);
	void savePsiPhi(std::string path);
	void gpu(int aSteps, int vSteps, float minAngle, float maxAngle,
			float minVelocity, float maxVelocity, int minObservations);
	void cpu(int aSteps, int vSteps, float minAngle, float maxAngle,
			float minVelocity, float maxVelocity, int minObservations);
	void filterResults(int minObservations);
	std::vector<trajRegion> regionSearch(float xVel, float yVel,
			float radius, float minLH, int minObservations);
	trajRegion& calculateLH(trajRegion& t);
	std::vector<float> observeTrajectory(
			trajRegion& t, std::vector<std::vector<RawImage>>& pooledImgs, int poolType);
	float findExtremeInRegion(float x, float y, int size,
			std::vector<RawImage>& pooledImgs, int poolType);
	int biggestFit(int x, int y, int maxX, int maxY); // inline?
	float readPixelDepth(int depth, int x, int y, std::vector<RawImage>& pooledImgs);
	std::vector<trajRegion>& calculateLHBatch(std::vector<trajRegion>& tlist);
	std::vector<trajRegion> subdivide(trajRegion& t);
	std::vector<trajRegion>& filterBounds(std::vector<trajRegion>& tlist,
			float xVel, float yVel, float ft, float radius);
	float squareSDF(float scale, float centerX, float centerY,
			float pointX, float pointY);
	std::vector<trajRegion>& filterLH(std::vector<trajRegion>& tlist, float minLH, int minObs);
	//std::vector<float>& filterOutliers(std::vector<float>& obs);
	float pixelExtreme(float pixel, float prev, int poolType);
	float maxMasked(float pixel, float previousMax);
	float minMasked(float pixel, float previousMin);
	trajectory convertTraj(trajRegion& t);
	std::vector<RawImage> createStamps(trajectory t, int radius, std::vector<RawImage*> imgs);
    std::vector<float> createCurves(trajectory t, std::vector<RawImage*> imgs);
	RawImage stackedStamps(trajectory t, int radius, std::vector<RawImage*> imgs);
	RawImage stackedScience(trajRegion& t, int radius);
	std::vector<RawImage> scienceStamps(trajRegion& t, int radius);
	std::vector<RawImage> psiStamps(trajRegion& t, int radius);
	std::vector<RawImage> phiStamps(trajRegion& t, int radius);
	RawImage stackedScience(trajectory& t, int radius);
	std::vector<RawImage> scienceStamps(trajectory& t, int radius);
	std::vector<RawImage> psiStamps(trajectory& t, int radius);
	std::vector<RawImage> phiStamps(trajectory& t, int radius);
    std::vector<float> psiCurves(trajectory& t);
    std::vector<float> phiCurves(trajectory& t);
	std::vector<trajectory> getResults(int start, int end);
	std::vector<RawImage>& getPsiImages();
    std::vector<RawImage>& getPhiImages();
    std::vector<std::vector<RawImage>>& getPsiPooled();
    std::vector<std::vector<RawImage>>& getPhiPooled();
 	void clearPsiPhi();
	void saveResults(std::string path, float fraction);
	void setDebug(bool d) { debugInfo = d; };
	virtual ~KBMOSearch() {};

private:
	void search(bool useGpu, int aSteps, int vSteps, float minAngle,
			float maxAngle, float minVelocity, float maxVelocity, int minObservations);
	std::vector<trajRegion> resSearch(float xVel, float yVel,
			float radius, int minObservations, float minLH);
	std::vector<trajRegion> resSearchGPU(float xVel, float yVel,
			float radius, int minObservations, float minLH);
	void clearPooled();
	void preparePsiPhi();
	void poolAllImages();
	std::vector<std::vector<RawImage>>& poolSet(
			std::vector<RawImage> imagesToPool,
			std::vector<std::vector<RawImage>>& destination, short mode);
	std::vector<RawImage> poolSingle(std::vector<RawImage>& mip, RawImage& img, short mode);
	void repoolArea(trajRegion& t);
	void cpuConvolve();
	void gpuConvolve();
	void removeObjectFromImages(trajRegion& t);
	void saveImages(std::string path);
	void createSearchList(int angleSteps, int veloctiySteps, float minAngle,
			float maxAngle, float minVelocity, float maxVelocity);
	void createInterleavedPsiPhi();
	void cpuSearch(int minObservations);
	void gpuSearch(int minObservations);
	void sortResults();
	void startTimer(std::string message);
	void endTimer();
	long int totalPixelsRead;
	long int regionsMaxed;
	long int searchRegionsBounded;
	long int individualEval;
	long long nodesProcessed;
	unsigned maxResultCount;
	bool psiPhiGenerated;
	bool debugInfo;
	std::chrono::time_point<std::chrono::system_clock> tStart, tEnd;
	std::chrono::duration<double> tDelta;
	ImageStack stack;
	PointSpreadFunc psf;
	PointSpreadFunc psfSQ;
	std::vector<trajectory> searchList;
	std::vector<RawImage> psiImages;
	std::vector<RawImage> phiImages;
	std::vector<std::vector<RawImage>> pooledPsi;
	std::vector<std::vector<RawImage>> pooledPhi;
	std::vector<float> interleavedPsiPhi;
	std::vector<trajectory> results;

};

} /* namespace kbmod */

#endif /* KBMODSEARCH_H_ */
