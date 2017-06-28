/*
 * KBMOSearch.h
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#ifndef KBMODSEARCH_H_
#define KBMODSEARCH_H_

#include "common.h"
#include "ImageStack.h"

class KBMOSearch {
public:
	KBMOSearch(PointSpreadFunc PSF);
	void gpu(ImageStack stack, std::string resultsPath,
			float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	void cpu(ImageStack stack, std::string resultsPath,
			float minAngle, float maxAngle, float minVelocity, float maxVelocity);
	virtual ~KBMOSearch();
private:
	void createPhiPSF();
	void cpuConvolve();
	void gpuConvolve();
	void createPsiPhi();
	void saveImages();
	void createsearchList();
	void cpuSearch();
	void gpuSearch();
	void sortResults();
	void saveResults();
	PointSpreadFunc psf;
	PointSpreadFunc psfSQ;
	std::vector<trajectories> searchList;
	std::vector<std::vector<float>> psiImages;
	std::vector<std::vector<float>> phiImages;
	std::vector<float> interleavedPsiPhi;
	std::vector<trajectory> results;
	bool savePsiPhi;

};

#endif /* KBMODSEARCH_H_ */
