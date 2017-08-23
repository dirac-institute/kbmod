/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace kbmod {

KBMOSearch::KBMOSearch(ImageStack imstack, PointSpreadFunc PSF) :
		psf(PSF), psfSQ(PSF.getStdev()), stack(imstack)
{
	psfSQ.squarePSF();
	saveResultsFlag = true;
}

void KBMOSearch::gpu(
		int aSteps, int vSteps, float minAngle, float maxAngle,
		float minVelocity, float maxVelocity, int minObservations)
{
	search(true, aSteps, vSteps, minAngle,
			maxAngle, minVelocity, maxVelocity, minObservations);
}
void KBMOSearch::cpu(
		int aSteps, int vSteps, float minAngle, float maxAngle,
		float minVelocity, float maxVelocity, int minObservations)
{
	search(false, aSteps, vSteps, minAngle,
			maxAngle, minVelocity, maxVelocity, minObservations);
}

void KBMOSearch::savePsiPhi(std::string path)
{
	preparePsiPhi();
	gpuConvolve();
	saveImages(path);
}

void KBMOSearch::search(bool useGpu, int aSteps, int vSteps, float minAngle, float maxAngle,
		float minVelocity, float maxVelocity, int minObservations)
{
	preparePsiPhi();
	useGpu ? gpuConvolve() : cpuConvolve();
	createSearchList(aSteps, vSteps, minAngle, maxAngle, minVelocity, maxVelocity);
	createInterleavedPsiPhi();
	results = std::vector<trajectory>(stack.getPPI()*RESULTS_PER_PIXEL);
	std::cout << "searching " << searchList.size() << " trajectories... " << std::flush;
	useGpu ? gpuSearch(minObservations) : cpuSearch(minObservations);
	std::cout << "Done.\n" << std::flush;
	// Free all but results?
	interleavedPsiPhi = std::vector<float>();
	sortResults();
}

void KBMOSearch::clearPsiPhi()
{
	psiImages = std::vector<RawImage>();
	phiImages = std::vector<RawImage>();
}

void KBMOSearch::preparePsiPhi()
{
	// Compute Phi and Psi from convolved images
	// while leaving masked pixels alone
	// Reinsert 0s for MASK_FLAG?
	clearPsiPhi();
	std::vector<LayeredImage> imgs = stack.getImages();
	for (unsigned i=0; i<stack.imgCount(); ++i)
	{
		float *sciArray = imgs[i].getSDataRef();
		float *varArray = imgs[i].getVDataRef();
		std::vector<float> currentPsi = std::vector<float>(stack.getPPI());
		std::vector<float> currentPhi = std::vector<float>(stack.getPPI());
		for (unsigned p=0; p<stack.getPPI(); ++p)
		{
			float varPix = varArray[p];
			if (varPix != MASK_FLAG)
			{
				currentPsi[p] = sciArray[p]/varPix;
				currentPhi[p] = 1.0/varPix;
			} else {
				currentPsi[p] = MASK_FLAG;
				currentPhi[p] = MASK_FLAG;
			}

		}
		psiImages.push_back(RawImage(stack.getWidth(), stack.getHeight(), currentPsi));
		phiImages.push_back(RawImage(stack.getWidth(), stack.getHeight(), currentPhi));
	}
}

void KBMOSearch::cpuConvolve()
{

}

void KBMOSearch::gpuConvolve()
{
	unsigned index = 0;
	for (auto& i : stack.getImages())
	{
		psiImages[index].convolve(psf);
		phiImages[index].convolve(psfSQ);
		index++;
	}
}

void KBMOSearch::saveImages(std::string path)
{
	for (unsigned i=0; i<stack.imgCount(); ++i)
	{
		std::string number = std::to_string(i);
		// Add leading zeros
		number = std::string(4 - number.length(), '0') + number;
		psiImages[i].saveToFile(path+"/psi/PSI"+number+".fits");
		phiImages[i].saveToFile(path+"/phi/PHI"+number+".fits");
	}
}

void KBMOSearch::createSearchList(int angleSteps, int velocitySteps,
		float minAngle, float maxAngle,
		float minVelocity, float maxVelocity)
{

		// const angleSteps and velocitySteps for now
		//const int angleSteps = 10;
		//const int velocitySteps = 10;
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
	interleavedPsiPhi = std::vector<float>(stack.imgCount()*stack.getPPI()*2);
	for (unsigned i=0; i<stack.imgCount(); ++i)
	{
		unsigned iImgPix = i*stack.getPPI()*2;
		float *psiRef = psiImages[i].getDataRef();
		float *phiRef = phiImages[i].getDataRef();
		for (unsigned p=0; p<stack.getPPI(); ++p)
		{
			unsigned iPix = p*2;
			interleavedPsiPhi[iImgPix+iPix]   = psiRef[p];
			interleavedPsiPhi[iImgPix+iPix+1] = phiRef[p];
		}
	}
	// Clear old psi phi buffers
	//clearPsiPhi();
}

void KBMOSearch::cpuSearch(int minObservations)
{

}

void KBMOSearch::gpuSearch(int minObservations)
{
	deviceSearch(searchList.size(), stack.imgCount(), minObservations,
			interleavedPsiPhi.size(), stack.getPPI()*RESULTS_PER_PIXEL,
			searchList.data(), results.data(), stack.getTimes().data(),
			interleavedPsiPhi.data(), stack.getWidth(), stack.getHeight());
}

std::vector<RawImage> KBMOSearch::getPsiImages() {
	return psiImages;
}

std::vector<RawImage> KBMOSearch::getPhiImages() {
	return phiImages;
}

void KBMOSearch::sortResults()
{
	__gnu_parallel::sort(results.begin(), results.end(),
			[](trajectory a, trajectory b) {
		return b.lh < a.lh;
	});
}

void KBMOSearch::filterResults(int minObservations)
{
	results.erase(
			std::remove_if(results.begin(), results.end(),
					std::bind([](trajectory t, int cutoff) {
						return t.obsCount<cutoff;
	}, std::placeholders::_1, minObservations)),
	results.end());
}

std::vector<trajectory> KBMOSearch::getResults(int start, int count){
	assert(start>=0 && start+count<results.size());
	return std::vector<trajectory>(results.begin()+start, results.begin()+start+count);
}

void KBMOSearch::saveResults(std::string path, float portion)
{
	std::ofstream file(path.c_str());
	if (file.is_open())
	{
		file << "# x y xv yv likelihood flux obs_count\n";
		int writeCount = int(portion*float(results.size()));
		for (int i=0; i<writeCount; ++i)
		{
			trajectory r = results[i];
			file << r.x << " " << r.y << " "
				 << r.xVel << " " << r.yVel << " " << r.lh
				 << " " << r.flux << " " << r.obsCount << "\n";
		}
		file.close();
	} else {
		std::cout << "Unable to open results file";
	}
}

} /* namespace kbmod */
