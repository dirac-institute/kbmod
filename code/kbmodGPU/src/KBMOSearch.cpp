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
	psiImages = std::vector<RawImage>();
	phiImages = std::vector<RawImage>();
}

void KBMOSearch::preparePsiPhi()
{
	// Reinsert 0s for MASK_FLAG?
	clearPsiPhi();
	std::vector<LayeredImage> imgs = stack->getImages();
	for (unsigned i=0; i<stack->imgCount(); ++i)
	{
		float *sciArray = imgs[i].getSDataRef();
		float *varArray = imgs[i].getVDataRef();
		std::vector<float> tempPsi = std::vector<float>(stack->getPPI());
		std::vector<float> tempPhi = std::vector<float>(stack->getPPI());
		for (unsigned p=0; p<stack->getPPI(); ++p)
		{
			float varPix = varArray[p];
			if (varPix != MASK_FLAG)
			{
				tempPsi[p] = sciArray[p]/varPix;
				tempPhi[p] = 1.0/varPix;
			} else {
				tempPsi[p] = MASK_FLAG;
				tempPhi[p] = MASK_FLAG;
			}

		}
		psiImages.push_back(RawImage(stack->getWidth(), stack->getHeight(), tempPsi.data()));
		phiImages.push_back(RawImage(stack->getWidth(), stack->getHeight(), tempPhi.data()));
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
		deviceConvolve(psiImages[index].getDataRef(), psiImages[index].getDataRef(),
					   stack->getDimensions(), psf);
		deviceConvolve(phiImages[index].getDataRef(), phiImages[index].getDataRef(),
					   stack->getDimensions(), psfSQ);
		index++;
	}
}

void KBMOSearch::saveImages(std::string path)
{
	for (unsigned i=0; i<stack->imgCount(); ++i)
	{
		psiImages[i].saveToFile(path+"/psi/PSI"+std::to_string(i)+".fits");
		phiImages[i].saveToFile(path+"/phi/PHI"+std::to_string(i)+".fits");
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
		float *psiRef = psiImages[i].getDataRef();
		float *phiRef = phiImages[i].getDataRef();
		for (unsigned p=0; p<stack->getPPI(); ++p)
		{
			unsigned iPix = p*2;
			interleavedPsiPhi[iImgPix+iPix]   = psiRef[p];
			interleavedPsiPhi[iImgPix+iPix+1] = phiRef[p];
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
