/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace kbmod {

KBMOSearch::KBMOSearch(ImageStack *imstack, PointSpreadFunc *PSF) {
	stack = imstack;
	psf = PSF;
	psfSQ = new PointSpreadFunc(psf->getStdev());
	psfSQ->squarePSF();
	savePsiPhi = false;
	saveResultsFlag = true;
}

void KBMOSearch::gpu(
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{
	search(true, minAngle, maxAngle, minVelocity, maxVelocity);
}
void KBMOSearch::cpu(
		float minAngle, float maxAngle, float minVelocity, float maxVelocity)
{
	search(false, minAngle, maxAngle, minVelocity, maxVelocity);
}

void KBMOSearch::search(bool useGpu, float minAngle, float maxAngle,
		float minVelocity, float maxVelocity)
{
	preparePsiPhi();
	useGpu ? gpuConvolve() : cpuConvolve();
	if (imageOutPath.length()>0)	saveImages(imageOutPath);
	createSearchList(minAngle, maxAngle, minVelocity, maxVelocity);
	createInterleavedPsiPhi();
	results = std::vector<trajectory>(stack->getPPI()*RESULTS_PER_PIXEL);
	std::cout << "searching " << searchList.size() << " trajectories... " << std::flush;
	useGpu ? gpuSearch() : cpuSearch();
	std::cout << "Done.\n" << std::flush;
	// Free all but results?
	sortResults();
}

void KBMOSearch::imageSaveLocation(std::string path)
{
	imageOutPath = path;
}

void KBMOSearch::clearPsiPhi()
{
	psiImages = std::vector<std::vector<float>>(stack->imgCount());
	phiImages = std::vector<std::vector<float>>(stack->imgCount());
}

void KBMOSearch::preparePsiPhi()
{
	// Reinsert 0s for MASK_FLAG?
	clearPsiPhi();
	std::vector<RawImage> imgs = stack->getImages();
	for (unsigned i=0; i<stack->imgCount(); ++i)
	{
		float *sciArray = imgs[i].getSDataRef();
		float *varArray = imgs[i].getVDataRef();
		psiImages[i] = std::vector<float>(stack->getPPI());
		phiImages[i] = std::vector<float>(stack->getPPI());
		for (unsigned p=0; p<stack->getPPI(); ++p)
		{
			float varPix = varArray[p];
			if (varPix != MASK_FLAG)
			{
				psiImages[i][p] = sciArray[p]/varPix;
				phiImages[i][p] = 1.0/varPix;
			} else {
				psiImages[i][p] = MASK_FLAG;
				phiImages[i][p] = MASK_FLAG;
			}

		}
		imgs[i].freeLayers();
	}
}

void KBMOSearch::cpuConvolve()
{

}

void KBMOSearch::gpuConvolve()
{
	unsigned index = 0;
	for (auto& i : stack->getImages())
	{
		deviceConvolve(psiImages[index].data(), psiImages[index].data(),
					   stack->getDimensions(), psf);
		deviceConvolve(phiImages[index].data(), phiImages[index].data(),
					   stack->getDimensions(), psfSQ);
		index++;
	}
}

void KBMOSearch::saveImages(std::string path)
{
	for (unsigned i=0; i<stack->imgCount(); ++i)
	{
		RawImage::writeFitsImg((path+"/psi/PSI"+std::to_string(i)+".fits"),
				psiImages[i].data(), stack->getDimensions(), stack->getPPI());
		RawImage::writeFitsImg((path+"/phi/PHI"+std::to_string(i)+".fits"),
				phiImages[i].data(), stack->getDimensions(), stack->getPPI());
	}
}

void KBMOSearch::createSearchList(float minAngle, float maxAngle,
		float minVelocity, float maxVelocity)
{

		// const angleSteps and velocitySteps for now
		const int angleSteps = 10;
		const int velocitySteps = 10;
		std::vector<float> angles(angleSteps);
		float aStepSize = (maxAngle-minAngle)/float(angleSteps);
		for (int i=0; i<angleSteps; ++i)
		{
			angles[i] = minAngle+float(i)*aStepSize;
		}

		std::vector<float> velocities(velocitySteps);
		float vStepSize = (maxVelocity-minVelocity)/float(velocitySteps);
		for (int i=0; i<velocitySteps; ++i)
		{
			velocities[i] = minVelocity+float(i)*vStepSize;
		}

		int trajCount = angleSteps*velocitySteps;
		searchList = std::vector<trajectory>(trajCount);
		for (int a=0; a<angleSteps; ++a)
		{
			for (int v=0; v<velocitySteps; ++v)
			{
				searchList[a*velocitySteps+v].xVel = cos(angles[a])*velocities[v];
				searchList[a*velocitySteps+v].yVel = sin(angles[a])*velocities[v];
			}
		}
}

void KBMOSearch::createInterleavedPsiPhi()
{
	interleavedPsiPhi = std::vector<float>(stack->imgCount()*stack->getPPI()*2);
	for (unsigned i=0; i<stack->imgCount(); ++i)
	{
		unsigned iImgPix = i*stack->getPPI()*2;
		for (unsigned p=0; p<stack->getPPI(); ++p)
		{
			unsigned iPix = p*2;
			interleavedPsiPhi[iImgPix+iPix]   = psiImages[i][p];
			interleavedPsiPhi[iImgPix+iPix+1] = phiImages[i][p];
		}
	}
	// Clear old psi phi buffers
	clearPsiPhi();
}

void KBMOSearch::cpuSearch()
{

}

void KBMOSearch::gpuSearch()
{
	deviceSearch(searchList.size(), stack->imgCount(), interleavedPsiPhi.size(),
			stack->getPPI()*RESULTS_PER_PIXEL, searchList.data(),
			results.data(), stack->getTimes().data(),
			interleavedPsiPhi.data(), stack->getDimensions());
}

void KBMOSearch::sortResults()
{
	__gnu_parallel::sort(results.begin(), results.end(),
			[](trajectory a, trajectory b) {
		return b.lh < a.lh;
	});
}

void KBMOSearch::saveResults(std::string path, float portion)
{

	for (int i=0; i<500; ++i)
	{
		trajectory r = results[i];
		std::cout << "x: " << r.x << " y: " << r.y << " xv: "
				<< r.xVel << " yv: " << r.yVel << " lh: " << r.lh
				<< " flux: " << r.flux << "\n";
	}
	FILE *resultsFile = fopen(path.c_str(), "w");
	fwrite(&results[0], sizeof(results), static_cast<int>(results.size()/10/*portion*/), resultsFile);
	fclose(resultsFile);
}

KBMOSearch::~KBMOSearch() {
	delete psfSQ;
}

} /* namespace kbmod */
