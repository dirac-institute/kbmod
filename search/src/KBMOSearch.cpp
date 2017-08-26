/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace kbmod {

KBMOSearch::KBMOSearch(ImageStack imstack, PointSpreadFunc PSF) :
		psf(PSF), psfSQ(PSF.getStdev()), stack(imstack), pooledPsi(), pooledPhi()
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

void KBMOSearch::search(bool useGpu, int aSteps, int vSteps, float minAngle,
		float maxAngle, float minVelocity, float maxVelocity, int minObservations)
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

void KBMOSearch::multiResSearch(float xVel, float yVel,
		float radius, int minObservations, float minLH)
{
	preparePsiPhi();
	gpuConvolve();
	poolAllImages();
	resSearch(xVel, yVel, radius, minObservations, minLH);
	clearPooled();
}

void KBMOSearch::clearPsiPhi()
{
	psiImages = std::vector<RawImage>();
	phiImages = std::vector<RawImage>();
}

void KBMOSearch::clearPooled()
{
	pooledPsi = std::vector<std::vector<RawImage>>();
	pooledPhi = std::vector<std::vector<RawImage>>();
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

void KBMOSearch::poolAllImages()
{
	poolSet(psiImages, pooledPsi, POOL_MAX);
	poolSet(phiImages, pooledPhi, POOL_MIN);
}

void KBMOSearch::poolSet(std::vector<RawImage> imagesToPool,
		std::vector<std::vector<RawImage>> destination, short mode)
{
	for (auto& i : imagesToPool) {
		std::vector<RawImage> pooled = std::vector<RawImage>();
		poolSingle(pooled, i, mode);
		destination.push_back(pooled);
	}
}

void KBMOSearch::poolSingle(std::vector<RawImage> mip, RawImage img, short mode)
{
	mip.push_back(img);
	RawImage current = img;
	while (current.getPPI() > 1) {
		current = current.pool(mode);
		mip.push_back(current);
	}
}

void KBMOSearch::cpuConvolve()
{

}

void KBMOSearch::gpuConvolve()
{
	for (int i=0; i<stack.imgCount(); ++i)
	{
		psiImages[i].convolve(psf);
		phiImages[i].convolve(psfSQ);
	}
}

void KBMOSearch::saveImages(std::string path)
{
	for (int i=0; i<stack.imgCount(); ++i)
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

void KBMOSearch::resSearch(float xVel, float yVel,
		float radius, int minObservations, float minLH)
{
	int maxDepth = pooledPsi[0].size();
	assert(maxDepth>0 && maxDepth < 127);
	dtraj root = {0,0,0,0, maxDepth, 0, 0.0};
	calculateLH({root});
	// A function to sort trajectories
	auto cmpLH = [](dtraj a, dtraj b) { return a.likelihood < b.likelihood; };
	std::priority_queue<dtraj, std::vector<dtraj>, decltype(cmpLH)> candidates(cmpLH);
	candidates.push(root);
	while (candidates.size() > 0)
	{
		dtraj t = candidates.pop();
		if (t.depth==0) {
			std::cout << "ix: " << t.ix << " iy: " << t.iy << " lh: " t.likelihood;
			return;
		} else {
			std::vector<dtraj> sublist = subdivide(t);
			filterBounds(sublist, xVel, yVel, radius);
			calculateLH(sublist);
			filterLH(sublist, minLH, minObservations);
			for (auto& nt : sublist) candidates.push(nt);
		}
	}
}

std::vector<dtraj> KBMOSearch::subdivide(dtraj t)
{
	unsigned char nDepth = t.depth-1;
	std::vector<dtraj> children(16);
	children[0]  = {t.ix*2,  t.iy*2,  t.fx*2,  t.fy*2,  nDepth, 0, 0.0};
	children[1]  = {t.ix*2+1,t.iy*2,  t.fx*2,  t.fy*2,  nDepth, 0, 0.0};
	children[2]  = {t.ix*2  ,t.iy*2+1,t.fx*2,  t.fy*2,  nDepth, 0, 0.0};
	children[3]  = {t.ix*2+1,t.iy*2+1,t.fx*2,  t.fy*2,  nDepth, 0, 0.0};
	children[4]  = {t.ix*2  ,t.iy*2,  t.fx*2+1,t.fy*2,  nDepth, 0, 0.0};
	children[5]  = {t.ix*2+1,t.iy*2,  t.fx*2+1,t.fy*2,  nDepth, 0, 0.0};
	children[6]  = {t.ix*2  ,t.iy*2+1,t.fx*2+1,t.fy*2,  nDepth, 0, 0.0};
	children[7]  = {t.ix*2+1,t.iy*2+1,t.fx*2+1,t.fy*2,  nDepth, 0, 0.0};
	children[8]  = {t.ix*2  ,t.iy*2,  t.fx*2,  t.fy*2+1,nDepth, 0, 0.0};
	children[9]  = {t.ix*2+1,t.iy*2,  t.fx*2,  t.fy*2+1,nDepth, 0, 0.0};
	children[10] = {t.ix*2  ,t.iy*2+1,t.fx*2,  t.fy*2+1,nDepth, 0, 0.0};
	children[11] = {t.ix*2+1,t.iy*2+1,t.fx*2,  t.fy*2+1,nDepth, 0, 0.0};
	children[12] = {t.ix*2  ,t.iy*2,  t.fx*2+1,t.fy*2+1,nDepth, 0, 0.0};
	children[13] = {t.ix*2+1,t.iy*2,  t.fx*2+1,t.fy*2+1,nDepth, 0, 0.0};
	children[14] = {t.ix*2  ,t.iy*2+1,t.fx*2+1,t.fy*2+1,nDepth, 0, 0.0};
	children[15] = {t.ix*2+1,t.iy*2+1,t.fx*2+1,t.fy*2+1,nDepth, 0, 0.0};

	return children;
}

void KBMOSearch::filterBounds(
		std::vector<dtraj> tlist, float xVel, float yVel, float radius)
{

}
void KBMOSearch::filterLH(std::vector<dtraj> tlist, float minLH, int minObs)
{

}

void KBMOSearch::calculateLH(std::vector<dtraj> tlist)
{
	std::for_each(tlist.begin(), tlist.end(), calculateLH);
}

void KBMOSearch::calculateLH(dtraj t)
{
	float psiSum = 0.0;
	float phiSum = 0.0;
	std::vector<float> times = stack.getTimes();
	float endTime = times.back();
	float xv = static_cast<float>(t.fx-t.ix)/endTime;
	float yv = static_cast<float>(t.fy-t.iy)/endTime;
	for (int i=0; i<stack.imgCount(); ++i)
	{
		int xp = static_cast<int>(static_cast<float>(t.ix) + times[i] * xv + 0.5);
		int yp = static_cast<int>(static_cast<float>(t.iy) + times[i] * yv + 0.5);

		float tempPsi = pooledPsi[i][t.depth].getPixel(xp,yp);
		if (tempPsi == MASK_FLAG) continue;
		psiSum += tempPsi;
		phiSum += pooledPhi[i][t.depth].getPixel(xp,yp);
		t.obs_count++;
	}
	assert(phiSum>0.0);
	t.likelihood = psiSum/sqrt(phiSum);
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
