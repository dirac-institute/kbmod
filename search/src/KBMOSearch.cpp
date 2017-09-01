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
		float radius, float minLH, int minObservations)
{
	preparePsiPhi();
	gpuConvolve();
	clearPooled();
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
	for (int i=0; i<stack.imgCount(); ++i)
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
	pooledPsi = poolSet(psiImages, pooledPsi, POOL_MAX);
	pooledPhi = poolSet(phiImages, pooledPhi, POOL_MIN);
}

std::vector<std::vector<RawImage>> KBMOSearch::poolSet(
		std::vector<RawImage> imagesToPool,
		std::vector<std::vector<RawImage>> destination, short mode)
{
	for (auto& i : imagesToPool) {
		std::vector<RawImage> pooled;
		pooled = poolSingle(pooled, i, mode);
		destination.push_back(pooled);
	}
	return destination;
}

std::vector<RawImage> KBMOSearch::poolSingle(
		std::vector<RawImage> mip, RawImage img, short mode)
{
	mip.push_back(img);
	RawImage current = img;
	while (current.getPPI() > 1) {
		current = current.pool(mode);
		mip.push_back(current);
	}
	return mip;
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
	for (int i=0; i<stack.imgCount(); ++i)
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
	float finalTime = stack.getTimes().back();
	assert(maxDepth>0 && maxDepth < 127);
	dtraj root = {0,0,0,0, static_cast<char>(maxDepth), 0, 0.0};
	calculateLH(root);
	std::cout << "evaluating root:" << " depth: " << static_cast<int>(root.depth)
			<<" ix: " << root.ix << " iy: " << root.iy << " lh: " << root.likelihood << "\n";
	// A function to sort trajectories
	auto cmpLH = [](dtraj a, dtraj b) { return a.likelihood < b.likelihood; };
	std::priority_queue<dtraj, std::vector<dtraj>, decltype(cmpLH)> candidates(cmpLH);
	candidates.push(root);
	while (!candidates.empty())
	{
		dtraj t = candidates.top();
		std::cout << "evaluating:" << " depth: " << static_cast<int>(t.depth)
				<<" ix: " << t.ix << " iy: " << t.iy << " lh: " << t.likelihood << "\n";

		candidates.pop();
		if (t.depth==0) {
			std::cout << "ix: " << t.ix << " iy: " << t.iy << " lh: " << t.likelihood;
			return;
		} else {
			std::vector<dtraj> sublist = subdivide(t);
			sublist = filterBounds(sublist, xVel, yVel, finalTime, radius);
			sublist = calculateLHBatch(sublist);
			sublist = filterLH(sublist, minLH, minObservations);
			for (auto& nt : sublist) candidates.push(nt);
		}
	}
}

std::vector<dtraj> KBMOSearch::subdivide(dtraj t)
{
	char nDepth = t.depth-1;
	std::vector<dtraj> children(16);
	// Must explicitly cast to short after addition to
	// avoid compiler warnings
	auto c = [](short x){ return static_cast<short>(x); };
	short nix = c(t.ix*2);
	short niy = c(t.iy*2);
	short nfx = c(t.fx*2);
	short nfy = c(t.fy*2);
	const short s = 1;
	children[0]  = {  nix,     niy,     nfx,     nfy,   nDepth, 0, 0.0};
	children[1]  = {c(nix+s),  niy,     nfx,     nfy,   nDepth, 0, 0.0};
	children[2]  = {  nix,   c(niy+s),  nfx,     nfy,   nDepth, 0, 0.0};
	children[3]  = {c(nix+s),c(niy+s),  nfx,     nfy,   nDepth, 0, 0.0};
	children[4]  = {  nix,     niy,   c(nfx+s),  nfy,   nDepth, 0, 0.0};
	children[5]  = {c(nix+s),  niy,   c(nfx+s),  nfy,   nDepth, 0, 0.0};
	children[6]  = {  nix,   c(niy+s),c(nfx+s),  nfy,   nDepth, 0, 0.0};
	children[7]  = {c(nix+s),c(niy+s),c(nfx+s),  nfy,   nDepth, 0, 0.0};
	children[8]  = {  nix,     niy,     nfx,   c(nfy+s),nDepth, 0, 0.0};
	children[9]  = {c(nix+s),  niy,     nfx,   c(nfy+s),nDepth, 0, 0.0};
	children[10] = {  nix,   c(niy+s),  nfx,   c(nfy+s),nDepth, 0, 0.0};
	children[11] = {c(nix+s),c(niy+s),  nfx,   c(nfy+s),nDepth, 0, 0.0};
	children[12] = {  nix,     niy,   c(nfx+s),c(nfy+s),nDepth, 0, 0.0};
	children[13] = {c(nix+s),  niy,   c(nfx+s),c(nfy+s),nDepth, 0, 0.0};
	children[14] = {  nix,   c(niy+s),c(nfx+s),c(nfy+s),nDepth, 0, 0.0};
	children[15] = {c(nix+s),c(niy+s),c(nfx+s),c(nfy+s),nDepth, 0, 0.0};

	return children;
}

std::vector<dtraj> KBMOSearch::filterBounds(std::vector<dtraj> tlist,
		float xVel, float yVel, float ft, float radius)
{

	tlist.erase(
			std::remove_if(tlist.begin(), tlist.end(),
				std::bind([](dtraj t,
				float xv, float yv, float finalT, float rad) {
					unsigned int iscale = 1;
					// 2 raised to the depth power
					iscale = iscale << t.depth;
					float scale = static_cast<float>(iscale);
					float centerX = scale*(static_cast<float>(t.fx)+0.5);
					float centerY = scale*(static_cast<float>(t.fy)+0.5);
					float posX = scale*static_cast<float>(t.ix+0.5)+xv*finalT;
					float posY = scale*static_cast<float>(t.iy+0.5)*yv*finalT;
					// 2D box signed distance function
					float dx = posX-centerX;
					float dy = posY-centerY;
					float size = scale*0.5;
					float xn = abs(dx)-size;
					float yn = abs(dy)-size;
					float xk = std::min(xn,0.0f);
					float yk = std::min(yn,0.0f);
					float xm = std::max(xn,0.0f);
					float ym = std::max(yn,0.0f);
					float dist = sqrt(xm*xm+ym*ym) + std::max(xk,yk);
					return (dist - rad) > 0.0;
				}, std::placeholders::_1, xVel, yVel, ft, radius)),
	tlist.end());
	return tlist;
}

std::vector<dtraj> KBMOSearch::filterLH(std::vector<dtraj> tlist, float minLH, int minObs)
{
	tlist.erase(
			std::remove_if(tlist.begin(), tlist.end(),
					std::bind([](dtraj t, int cutoff, float mLH) {
						return t.obs_count<cutoff && t.likelihood<mLH;
	}, std::placeholders::_1, minObs, minLH)),
	tlist.end());
	return tlist;
}

std::vector<dtraj> KBMOSearch::calculateLHBatch(std::vector<dtraj> tlist)
{
	for (auto& t : tlist) calculateLH(t);
	return tlist;
	/*
	std::for_each(tlist.begin(), tlist.end(),
			std::bind([](dtraj t, KBMOSearch search)
					{ search.calculateLH(t); }, std::placeholders::_1, this));
	*/
}

dtraj KBMOSearch::calculateLH(dtraj t)
{
	float psiSum = 0.0;
	float phiSum = 0.0;
	std::vector<float> times = stack.getTimes();
	float endTime = times.back();
	float xv = static_cast<float>(t.fx-t.ix)/endTime;
	float yv = static_cast<float>(t.fy-t.iy)/endTime;
	for (int i=0; i<stack.imgCount(); ++i)
	{
		// Read from region rather than single pixel
		if (t.depth > 0) {
			float x = static_cast<float>(t.ix) + times[i] * xv;
			float y = static_cast<float>(t.iy) + times[i] * yv;
			int size = 1 << t.depth;
			float tempPsi = findExtremeInRegion(x, y, size, pooledPsi[i], POOL_MAX );
			std::cout << "in region x: " << x << " y: " << y << " size: " << size << " maxPsi: " << tempPsi << "\n";
			//float tempPsi = pooledPsi[i][t.depth].getPixel(xp,yp);
			if (tempPsi == MASK_FLAG) continue;
			psiSum += tempPsi;
			phiSum += findExtremeInRegion(x, y, size, pooledPhi[i], POOL_MIN );
			t.obs_count++;
		} else {
			int xp = static_cast<int>(static_cast<float>(t.ix) + times[i] * xv + 0.5);
			int yp = static_cast<int>(static_cast<float>(t.iy) + times[i] * yv + 0.5);
			float tempPsi = pooledPsi[i][t.depth].getPixel(xp,yp);
			if (tempPsi == MASK_FLAG) continue;
			psiSum += tempPsi;
			phiSum += pooledPhi[i][t.depth].getPixel(xp,yp);
			t.obs_count++;
		}
	}
	//assert(phiSum>0.0);
	t.likelihood = phiSum > 0.0 ? psiSum/sqrt(phiSum) : MASK_FLAG;
	return t;
}

float KBMOSearch::findExtremeInRegion(float x, float y, int size, std::vector<RawImage> pooledImgs, int poolType)
{
	// parameter for # of depths smaller to look than "size"
	x *= static_cast<float>(size);
	y *= static_cast<float>(size);
	float s = static_cast<float>(size)*0.5;
	// lower left corner of region
	int lx = static_cast<int>(floor(x-s));
	int ly = static_cast<int>(floor(y-s));
	// upper right corner of region
	int hx = static_cast<int>(ceil(x+s));
	int hy = static_cast<int>(ceil(y+s));
	float regionExtreme = poolType == POOL_MAX ? -FLT_MAX : FLT_MAX; // start opposite of goal
	int curY = ly;
	while (curY < hy) {
		int curX = lx;
		int rowLargest = 0;
		std::vector<int> rowSizes = std::vector<int>();
		while (curX < hx) {
			int optimalSize = biggestFit(curX, curY, hx, hy, size);
			//assert(rowLargest>0);
			float pix = readPixelDepth(optimalSize, curX, curY, pooledImgs);
			regionExtreme = pixelExtreme(pix, regionExtreme, poolType);
			if (optimalSize > rowLargest) {
				// Go back and read previous pixels up to size of largest
				int tempX = 0;
				for (auto& p : rowSizes) {
					int needToAdd = optimalSize/p;
					int startingHeight = rowLargest/p;
					// First one has already been maxed
					for (int i=startingHeight; i<needToAdd; i++) {
						float npix = readPixelDepth(p, tempX, i*p+curY, pooledImgs);
						regionExtreme = pixelExtreme(npix, regionExtreme, poolType);
					}
					tempX += p;
				}
				rowLargest = optimalSize;
			}
			curX += optimalSize;
			rowSizes.push_back(optimalSize);
		}
		curY += rowLargest;
	}
	return regionExtreme;
}

float KBMOSearch::pixelExtreme(float pixel, float prev, int poolType)
{
	return poolType == POOL_MAX ? maxMasked(pixel, prev) : minMasked(pixel, prev);
}

float KBMOSearch::maxMasked(float pixel, float previousMax)
{
	return pixel == MASK_FLAG ? previousMax : std::max(pixel, previousMax);
}

float KBMOSearch::minMasked(float pixel, float previousMin)
{
	return pixel == MASK_FLAG ? previousMin : std::min(pixel, previousMin);
}

float KBMOSearch::readPixelDepth(int size, int x, int y, std::vector<RawImage> pooledImgs)
{
	int depth = 0;
	// computer integer log2
	int tempLog = size;
	while (tempLog >>= 1) ++depth;
	float val = pooledImgs[depth].getPixel(x/size, y/size);
	std::cout << "attempting to read pixel depth: " << depth << " x: "
			<< x/size << " y: " << y/size << " val: " << val << " im width: "
			<< pooledImgs[depth].getWidth() << " y: " << pooledImgs[depth].getHeight() << "\n";
	return val;
}

int KBMOSearch::biggestFit(int x, int y, int maxX, int maxY, int maxSize) // inline?
{
	// check that maxSize is a power of two
	assert((maxSize&(-maxSize))==maxSize);
	int size = maxSize;
	while (x%size > 0 && y%size > 0 && x+size<maxX && y+size<maxY) {
		size /=2;
	}
	// should be at least 1
	assert(size>0);
	return size;
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
