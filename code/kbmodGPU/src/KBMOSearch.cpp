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
}

void KBMOSearch::gpu(ImageStack stack, std::string resultsPath,
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{
	createPhiPSF();
}
void KBMOSearch::cpu(ImageStack stack, std::string resultsPath,
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{

}

void KBMOSearch::createPhiPSF()
{

}

void KBMOSearch::cpuConvolve()
{

}

void KBMOSearch::gpuConvolve()
{

}

void KBMOSearch::createPsiPhi()
{

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

