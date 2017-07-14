/*
 * KBMOSearch.h
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#ifndef KBMODSEARCH_H_
#define KBMODSEARCH_H_

#include <parallel/algorithm>
//#include <fstream>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include "PointSpreadFunc.h"
#include "ImageStack.h"

namespace kbmod {

extern "C" void
deviceSearch(int trajCount, int imageCount, int psiPhiSize, int resultsCount,
			 trajectory * trajectoriesToSearch, trajectory *bestTrajects,
		     float *imageTimes, float *interleavedPsiPhi, long *dimensions);

class KBMOSearch {
public:
	KBMOSearch(ImageStack imstack, PointSpreadFunc PSF);
	void gpu(float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void cpu(float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void imageSaveLocation(std::string path);
	void saveResults(std::string path, float fraction);
	virtual ~KBMOSearch() {};

private:
	void search(bool useGpu,
			float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void createPSFSQ();
	void clearPsiPhi();
	void preparePsiPhi();
	void cpuConvolve();
	void gpuConvolve();
	void saveImages(std::string path);
	template<typename T>
	static void write_pod(std::ofstream& out, T& t);
	void createSearchList(float minAngle, float maxAngle,
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
	std::string imageOutPath;
	bool savePsiPhi;
	bool saveResultsFlag;

};

} /* namespace kbmod */

#endif /* KBMODSEARCH_H_ */
