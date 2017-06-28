/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

KBMOSearch::KBMOSearch(PointSpreadFunc PSF) {
	psf = PSF;
	savePsiPhi = false;
	saveResultsFlag = true;
}

void KBMOSearch::gpu(ImageStack stack, std::string resultsPath,
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{
	search(stack, resultsPath, true, minAngle, maxAngle, minVelocity, maxVelocity);
}
void KBMOSearch::cpu(ImageStack stack, std::string resultsPath,
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{
	search(stack, resultsPath, false, minAngle, maxAngle, minVelocity, maxVelocity);
}

void KBMOSearch::search(ImageStack stack, std::string resultsPath, bool useGpu,
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{
	createPhiPSF();
	useGpu ? gpuConvolve() : cpuConvolve();
	createPsiPhi();
	if (savePsiPhi)	saveImages();
	createsearchList();
	useGpu ? gpuSearch : cpuSearch();
	sortResults();
	if (saveResultsFlag) saveResults();
}

void KBMOSearch::createPhiPSF()
{
	psfSQ(psf.getStdev());
	psfSQ.squarePSF();
}

void KBMOSearch::cpuConvolve()
{

}

void KBMOSearch::gpuConvolve()
{

}

void KBMOSearch::createPsiPhi()
{
	// Reinsert 0s for MASK_FLAG?
}

void KBMOSearch::saveImages()
{

}

void KBMOSearch::createsearchList()
{

}

void KBMOSearch::cpuSearch()
{

}

void KBMOSearch::gpuSearch()
{

}

void KBMOSearch::sortResults()
{

}

void KBMOSearch::saveResults()
{

}

KBMOSearch::~KBMOSearch() {}

