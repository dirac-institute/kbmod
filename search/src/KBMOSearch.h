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
deviceSearch(int trajCount, int imageCount, int psiPhiSize, int resultsCount,
			 trajectory * trajectoriesToSearch, trajectory *bestTrajects,
		     float *imageTimes, float *interleavedPsiPhi, int width, int height);

class KBMOSearch {
public:
	KBMOSearch(ImageStack imstack, PointSpreadFunc PSF);
	void savePsiPhi(std::string path);
	void gpu(int aSteps, int vSteps, float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void cpu(int aSteps, int vSteps, float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void filterResults(int minObservations);
	std::vector<trajectory> getResults(int start, int end);
	void saveResults(std::string path, float fraction);
	virtual ~KBMOSearch() {};

private:
	void search(bool useGpu, int aSteps, int vSteps,
			float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void clearPsiPhi();
	void preparePsiPhi();
	void cpuConvolve();
	void gpuConvolve();
	void saveImages(std::string path);
	void createSearchList(int angleSteps, int veloctiySteps, float minAngle, float maxAngle,
			float minVelocity, float maxVelocity);
	void createInterleavedPsiPhi();
	void cpuSearch();
	void gpuSearch();
	void sortResults();
	ImageStack stack;
	PointSpreadFunc psf;
	PointSpreadFunc psfSQ;
	std::vector<trajectory> searchList;
	std::vector<RawImage> psiImages;
	std::vector<RawImage> phiImages;
	std::vector<float> interleavedPsiPhi;
	std::vector<trajectory> results;
	bool saveResultsFlag;

};

} /* namespace kbmod */

#endif /* KBMODSEARCH_H_ */
