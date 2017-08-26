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
#include <iostream>
#include <fstream>
//#include <stdio.h>
#include <assert.h>
#include "common.h"
#include "PointSpreadFunc.h"
#include "ImageStack.h"

namespace kbmod {

extern "C" void
deviceSearch(int trajCount, int imageCount, int minObservations, int psiPhiSize,
			 int resultsCount, trajectory * trajectoriesToSearch, trajectory *bestTrajects,
		     float *imageTimes, float *interleavedPsiPhi, int width, int height);

class KBMOSearch {
public:
	KBMOSearch(ImageStack imstack, PointSpreadFunc PSF);
	void savePsiPhi(std::string path);
	void gpu(int aSteps, int vSteps, float minAngle, float maxAngle,
			float minVelocity, float maxVelocity, int minObservations);
	void cpu(int aSteps, int vSteps, float minAngle, float maxAngle,
			float minVelocity, float maxVelocity, int minObservations);
	void filterResults(int minObservations);
	void multiResSearch(float xVel, float yVel, float radius, int minObservations);
	std::vector<trajectory> getResults(int start, int end);
	std::vector<RawImage> getPsiImages();
    std::vector<RawImage> getPhiImages();
	void saveResults(std::string path, float fraction);
	virtual ~KBMOSearch() {};

private:
	void search(bool useGpu, int aSteps, int vSteps, float minAngle,
			float maxAngle, float minVelocity, float maxVelocity, int minObservations);
	void clearPsiPhi();
	void preparePsiPhi();
	void poolAllImages();
	void poolSet(std::vector<RawImage> imagesToPool,
			std::vector<std::vector<RawImage>> destination, short mode);
	void poolSingle(std::vector<RawImage> mip, RawImage img, short mode);
	void cpuConvolve();
	void gpuConvolve();
	void saveImages(std::string path);
	void createSearchList(int angleSteps, int veloctiySteps, float minAngle,
			float maxAngle, float minVelocity, float maxVelocity);
	void createInterleavedPsiPhi();
	void cpuSearch(int minObservations);
	void gpuSearch(int minObservations);
	void sortResults();
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
	bool saveResultsFlag;

};

} /* namespace kbmod */

#endif /* KBMODSEARCH_H_ */
