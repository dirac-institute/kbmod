/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace kbmod {

KBMOSearch::KBMOSearch(ImageStack& imstack, PointSpreadFunc PSF) :
		psf(PSF), psfSQ(PSF), stack(imstack), pooledPsi(), pooledPhi()
{
	psfSQ.squarePSF();
	totalPixelsRead = 0;
	regionsMaxed = 0;
	maxResultCount = 100000;
	debugInfo = false;
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
	startTimer("Preparing psi and phi images");
	preparePsiPhi();
	endTimer();
	startTimer("Convolving images");
	useGpu ? gpuConvolve() : cpuConvolve();
	endTimer();
	createSearchList(aSteps, vSteps, minAngle, maxAngle, minVelocity, maxVelocity);
	startTimer("Creating interleaved psi/phi buffer");
	createInterleavedPsiPhi();
	endTimer();
	results = std::vector<trajectory>(stack.getPPI()*RESULTS_PER_PIXEL);
	if (debugInfo) std::cout <<
			searchList.size() << " trajectories... \n" << std::flush;
	startTimer("Searching");
	useGpu ? gpuSearch(minObservations) : cpuSearch(minObservations);
	endTimer();
	// Free all but results?
	interleavedPsiPhi = std::vector<float>();
	startTimer("Sorting results");
	sortResults();
	endTimer();
}

std::vector<trajRegion> KBMOSearch::regionSearch(
		float xVel, float yVel, float radius,
		float minLH, int minObservations)
{
	startTimer("Preparing psi and phi images");
	preparePsiPhi();
	endTimer();
	startTimer("Convolving images");
	gpuConvolve();
	endTimer();
	clearPooled();
	startTimer("Pooling images");
	poolAllImages();
	endTimer();
	startTimer("Searching regions");
	std::vector<trajRegion> res =
			resSearch(xVel, yVel, radius, minObservations, minLH);
	endTimer();
	if (debugInfo) {
		std::cout << totalPixelsRead <<
				" pixels read, computed bounds on "
				<< regionsMaxed << " regions for an average of "
				<< static_cast<float>(totalPixelsRead)/static_cast<float>(regionsMaxed)
				<< " pixels read per region\n";
	}
	//clearPooled();
	return res;
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
	// Reinsert 0s for NO_DATA?
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
			if (varPix != NO_DATA)
			{
				currentPsi[p] = sciArray[p]/varPix;
				currentPhi[p] = 1.0/varPix;
			} else {
				currentPsi[p] = NO_DATA;
				currentPhi[p] = NO_DATA;
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

std::vector<std::vector<RawImage>>& KBMOSearch::poolSet(
		std::vector<RawImage>& imagesToPool,
		std::vector<std::vector<RawImage>>& destination, short mode)
{
	for (auto& i : imagesToPool) {
		std::vector<RawImage> pooled;

		pooled = poolSingle(pooled, i, mode);
		destination.push_back(pooled);
	}
	return destination;
}

std::vector<RawImage> KBMOSearch::poolSingle(
		std::vector<RawImage>& mip, RawImage& img, short mode)
{
	mip.push_back(img);
	RawImage& current = img;
	while (current.getPPI() > 1) {
		current = current.pool(mode);
		mip.push_back(current);
	}
	return mip;
}

void KBMOSearch::repoolArea(trajRegion& t)
{
	// Repool small area of images after bright object
	// has been removed
	// This should probably be refactored in to multiple methods
	std::vector<float> times = stack.getTimes();
	float xv = (t.fx-t.ix)/times.back();
	float yv = (t.fy-t.iy)/times.back();
	for (unsigned i=0; i<pooledPsi.size(); ++i)
	{
		std::vector<RawImage>& cPsi = pooledPsi[i];
		std::vector<RawImage>& cPhi = pooledPhi[i];
		float x = t.ix+xv*times[i];
		float y = t.iy+yv*times[i];

		for (unsigned depth=1; depth<cPsi.size(); ++depth)
		{
			float scale = std::pow(2.0,static_cast<float>(depth));
			// Block psf dim * 2 to make sure all light is blocked
			int minX = floor( static_cast<float>(x-psf.getDim())/scale );
			int maxX = ceil(  static_cast<float>(x+psf.getDim())/scale );
			int minY = floor( static_cast<float>(y-psf.getDim())/scale );
			int maxY = ceil(  static_cast<float>(y+psf.getDim())/scale );
			for (int px=minX; px<=maxX; ++px)
			{
				for (int py=minY; py<=maxY; ++py)
				{
					float pixel;
					float nPsi = -FLT_MAX;
					pixel = cPsi[depth-1].getPixel(px*2,  py*2);
					nPsi = pixelExtreme(pixel, nPsi, POOL_MAX);
					pixel = cPsi[depth-1].getPixel(px*2+1,py*2);
					nPsi = pixelExtreme(pixel, nPsi, POOL_MAX);
					pixel = cPsi[depth-1].getPixel(px*2,  py*2+1);
					nPsi = pixelExtreme(pixel, nPsi, POOL_MAX);
					pixel = cPsi[depth-1].getPixel(px*2+1,py*2+1);
					nPsi = pixelExtreme(pixel, nPsi, POOL_MAX);
					cPsi[depth].setPixel(px,py, nPsi);

					float nPhi =  FLT_MAX;
					pixel = cPhi[depth-1].getPixel(px*2,  py*2);
					nPhi = pixelExtreme(pixel, nPhi, POOL_MIN);
					pixel = cPhi[depth-1].getPixel(px*2+1,py*2);
					nPhi = pixelExtreme(pixel, nPhi, POOL_MIN);
					pixel = cPhi[depth-1].getPixel(px*2,  py*2+1);
					nPhi = pixelExtreme(pixel, nPhi, POOL_MIN);
					pixel = cPhi[depth-1].getPixel(px*2+1,py*2+1);
					nPhi = pixelExtreme(pixel, nPhi, POOL_MIN);
					cPhi[depth].setPixel(px,py, nPhi);
				}
			}

		}
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

std::vector<trajRegion> KBMOSearch::resSearch(float xVel, float yVel,
		float radius, int minObservations, float minLH)
{
	int maxDepth = pooledPsi[0].size()-1;
	int minDepth = 0;
	float finalTime = stack.getTimes().back();
	assert(maxDepth>0 && maxDepth < 127);
	trajRegion root = {0.0,0.0,0.0,0.0, static_cast<short>(maxDepth), 0, 0.0, 0.0};
	root = calculateLH(root);
	std::vector<trajRegion> fResults;
	// A function to sort trajectories
	auto cmpLH = [](trajRegion a, trajRegion b)
			{ return a.likelihood < b.likelihood; };
	std::priority_queue<trajRegion, std::vector<trajRegion>,
		decltype(cmpLH)> candidates(cmpLH);
	candidates.push(root);
	while (!candidates.empty())
	{
		trajRegion t = candidates.top();
		assert(t.likelihood != NO_DATA);
		calculateLH(t);
		candidates.pop();
		if (t.likelihood < minLH || t.obs_count < minObservations)
			continue;
		if (t.likelihood<candidates.top().likelihood) {
			// if the new score is lower, push it back into the queue
			candidates.push(t);
			continue;
		}
		if (debugInfo) {
			std::cout << "\r                                             ";
			std::cout << "\rdepth: " << static_cast<int>(t.depth)
					  << " lh: " << t.likelihood << " queue size: "
					  << candidates.size() << std::flush;
		}
		if (t.depth==minDepth) {
			float s = std::pow(2.0, static_cast<float>(minDepth));
			t.ix *= s;
			t.iy *= s;
			t.fx *= s;
			t.fy *= s;
			// Remove the objects pixels from future searching
			// and make sure section of images are
			// repooled after object removal
			removeObjectFromImages(t);
			repoolArea(t);

			fResults.push_back(t);
			if (fResults.size() >= maxResultCount) break;
		} else {
			std::vector<trajRegion> sublist = subdivide(t);
			sublist = filterBounds(sublist, xVel, yVel, finalTime, radius);
			sublist = calculateLHBatch(sublist);
			sublist = filterLH(sublist, minLH, minObservations);
			for (auto& nt : sublist) candidates.push(nt);
		}
	}
	std::cout << std::endl;
	return fResults;
}

std::vector<trajRegion> KBMOSearch::subdivide(trajRegion& t)
{
	short nDepth = t.depth-1;
	std::vector<trajRegion> children(16);
	float nix = t.ix*2.0;
	float niy = t.iy*2.0;
	float nfx = t.fx*2.0;
	float nfy = t.fy*2.0;
	const float s = 1.0;
	children[0]  = { nix,    niy,   nfx,    nfy,  nDepth, 0, 0.0, 0.0 };
	children[1]  = { nix+s,  niy,   nfx,    nfy,  nDepth, 0, 0.0, 0.0 };
	children[2]  = { nix,    niy+s, nfx,    nfy,  nDepth, 0, 0.0, 0.0 };
	children[3]  = { nix+s,  niy+s, nfx,    nfy,  nDepth, 0, 0.0, 0.0 };
	children[4]  = { nix,    niy,   nfx+s,  nfy,  nDepth, 0, 0.0, 0.0 };
	children[5]  = { nix+s,  niy,   nfx+s,  nfy,  nDepth, 0, 0.0, 0.0 };
	children[6]  = { nix,    niy+s, nfx+s,  nfy,  nDepth, 0, 0.0, 0.0 };
	children[7]  = { nix+s,  niy+s, nfx+s,  nfy,  nDepth, 0, 0.0, 0.0 };
	children[8]  = { nix,    niy,   nfx,    nfy+s,nDepth, 0, 0.0, 0.0 };
	children[9]  = { nix+s,  niy,   nfx,    nfy+s,nDepth, 0, 0.0, 0.0 };
	children[10] = { nix,    niy+s, nfx,    nfy+s,nDepth, 0, 0.0, 0.0 };
	children[11] = { nix+s,  niy+s, nfx,    nfy+s,nDepth, 0, 0.0, 0.0 };
	children[12] = { nix,    niy,   nfx+s,  nfy+s,nDepth, 0, 0.0, 0.0 };
	children[13] = { nix+s,  niy,   nfx+s,  nfy+s,nDepth, 0, 0.0, 0.0 };
	children[14] = { nix,    niy+s, nfx+s,  nfy+s,nDepth, 0, 0.0, 0.0 };
	children[15] = { nix+s,  niy+s, nfx+s,  nfy+s,nDepth, 0, 0.0, 0.0 };

	return children;
}

std::vector<trajRegion> KBMOSearch::filterBounds(std::vector<trajRegion>& tlist,
		float xVel, float yVel, float ft, float radius)
{
	tlist.erase(
			std::remove_if(tlist.begin(), tlist.end(),
				std::bind([](trajRegion t, KBMOSearch* s,
				float xv, float yv, float finalT, float rad) {
					// 2 raised to the depth power
					float scale = std::pow(2.0, static_cast<float>(t.depth));
					float centerX = scale*(t.fx+0.5);
					float centerY = scale*(t.fy+0.5);
					float posX =    scale*(t.ix+0.5)+xv*finalT;
					float posY =    scale*(t.iy+0.5)+yv*finalT;
					// 2D box signed distance function
					float dist =          s->squareSDF(scale, centerX,
							centerY, posX-0.5*scale, posY+0.5*scale);
					dist = std::min(dist, s->squareSDF(scale, centerX,
							centerY, posX+0.5*scale, posY+0.5*scale));
					dist = std::min(dist, s->squareSDF(scale, centerX,
							centerY, posX-0.5*scale, posY-0.5*scale));
					dist = std::min(dist, s->squareSDF(scale, centerX,
							centerY, posX+0.5*scale, posY-0.5*scale));
					/*
					if (dist - rad > 0.0)
						std::cout << "cuttingBounds: " <<
					    t.ix << " " << t.iy <<
					    " | " << t.fx << " " <<
					    t.fy << "\n";
					*/
					return (dist - rad) > 0.0;
				}, std::placeholders::_1, this, xVel, yVel, ft, radius)),
	tlist.end());
	return tlist;
}

float KBMOSearch::squareSDF(float scale,
		float centerX, float centerY, float pointX, float pointY)
{
	float dx = pointX-centerX;
	float dy = pointY-centerY;
	float xn = std::abs(dx)-scale*0.5f;
	float yn = std::abs(dy)-scale*0.5f;
	float xk = std::min(xn,0.0f);
	float yk = std::min(yn,0.0f);
	float xm = std::max(xn,0.0f);
	float ym = std::max(yn,0.0f);
	return sqrt(xm*xm+ym*ym) + std::max(xk,yk);
}

std::vector<trajRegion> KBMOSearch::filterLH(
		std::vector<trajRegion>& tlist, float minLH, int minObs)
{
	tlist.erase(
			std::remove_if(tlist.begin(), tlist.end(),
					std::bind([](trajRegion t, int mObs, float mLH) {
						/*
						if (t.likelihood<mLH) {
							std::cout << "cuttingLH: " <<
							   t.ix << " " << t.iy <<
							   " | " << t.fx << " " <<
							   t.fy << " lh: " << t.likelihood << "\n";
						}
						if (t.obs_count<mObs) {
							std::cout << "cuttingObs: " <<
							   t.ix << " " << t.iy <<
							   " | " << t.fx << " " <<
							   t.fy << " lh: " << t.likelihood << "\n";
						}
						*/
						return t.obs_count<mObs || t.likelihood<mLH;
	}, std::placeholders::_1, minObs, minLH)),
	tlist.end());
	return tlist;
}

std::vector<trajRegion> KBMOSearch::calculateLHBatch(std::vector<trajRegion>& tlist)
{
	for (auto& t : tlist) t = calculateLH(t);
	return tlist;
}

trajRegion KBMOSearch::calculateLH(trajRegion& t)
{
	float psiSum = 0.0;
	float phiSum = 0.0;
	std::vector<float> times = stack.getTimes();
	float endTime = times.back();
	float xv = (t.fx-t.ix)/endTime;
	float yv = (t.fy-t.iy)/endTime;
	t.obs_count = 0;
	for (int i=0; i<stack.imgCount(); ++i)
	{
		// Read from region rather than single pixel
		if (t.depth > 0) {
			float x = t.ix+0.5 + times[i] * xv;
			float y = t.iy+0.5 + times[i] * yv;
			int size = 1 << static_cast<int>(t.depth);
			float tempPsi = findExtremeInRegion(x, y, size, pooledPsi[i], POOL_MAX );
			if (tempPsi == NO_DATA) continue;
			psiSum += tempPsi;
			phiSum += findExtremeInRegion(x, y, size, pooledPhi[i], POOL_MIN );
			t.obs_count++;
		} else {
			// Allow for fractional pixel coordinates
			float fractionalComp = std::pow(2.0, static_cast<float>(t.depth));
			float xp = fractionalComp*(t.ix + times[i] * xv); // +0.5;
			float yp = fractionalComp*(t.iy + times[i] * yv); // +0.5;
			int d = std::max(static_cast<int>(t.depth), 0);
			float tempPsi = pooledPsi[i][d].getPixelInterp(xp,yp);
			if (tempPsi == NO_DATA) continue;
			psiSum += tempPsi;
			phiSum += pooledPhi[i][d].getPixelInterp(xp,yp);
			t.obs_count++;
		}
	}
	//assert(phiSum>0.0);
	t.likelihood = phiSum > 0.0 ? psiSum/sqrt(phiSum) : NO_DATA;
	t.flux = phiSum > 0.0 ? psiSum/phiSum : NO_DATA;
	return t;
}

float KBMOSearch::findExtremeInRegion(float x, float y,
	int size, std::vector<RawImage>& pooledImgs, int poolType)
{
	regionsMaxed++;
	// check that maxSize is a power of two
	assert((size&(-size))==size);
	x *= static_cast<float>(size);
	y *= static_cast<float>(size);
	int sizeToRead = std::max(size/REGION_RESOLUTION, 1);
	int depth = 0;
	// computer integer log2
	int tempLog = sizeToRead;
	while (tempLog >>= 1) ++depth;
	float s = static_cast<float>(size)*0.5;
	// lower left corner of region
	int lx = static_cast<int>(floor(x-s));
	int ly = static_cast<int>(floor(y-s));
	// Round lower corner down to align larger pixel size
	lx = lx/sizeToRead;
	ly = ly/sizeToRead;
	// upper right corner of region
	int hx = static_cast<int>(ceil(x+s));
	int hy = static_cast<int>(ceil(y+s));
	// Round Upper corner up to align larger pixel size
	hx = (hx+sizeToRead-1)/sizeToRead;
	hy = (hy+sizeToRead-1)/sizeToRead;
	float regionExtreme =
			poolType == POOL_MAX ? -FLT_MAX : FLT_MAX; // start opposite of goal
	int curY = ly;
	while (curY < hy) {
		int curX = lx;
		while (curX < hx) {
			float pix = readPixelDepth(depth, curX, curY, pooledImgs);
			regionExtreme = pixelExtreme(pix, regionExtreme, poolType);
			curX++;
		}
		curY++;
	}
	if (regionExtreme == FLT_MAX || regionExtreme == -FLT_MAX)
		regionExtreme = NO_DATA;
	return regionExtreme;
}

float KBMOSearch::pixelExtreme(float pixel, float prev, int poolType)
{
	return poolType == POOL_MAX ? maxMasked(pixel, prev) : minMasked(pixel, prev);
}

float KBMOSearch::maxMasked(float pixel, float previousMax)
{
	return pixel == NO_DATA ? previousMax : std::max(pixel, previousMax);
}

float KBMOSearch::minMasked(float pixel, float previousMin)
{
	return pixel == NO_DATA ? previousMin : std::min(pixel, previousMin);
}

float KBMOSearch::readPixelDepth(int depth, int x, int y, std::vector<RawImage>& pooledImgs)
{
	totalPixelsRead++;
	return pooledImgs[depth].getPixel(x, y);
}

int KBMOSearch::biggestFit(int x, int y, int maxX, int maxY) // inline?
{
	int size = 1;//maxSize;
	while ((x%size == 0 && y%size == 0) && (x+size<=maxX && y+size<=maxY)) {
		size *= 2;
	}
	size /= 2;
	// should be at least 1
	assert(size>0);
	return size;
}

void KBMOSearch::removeObjectFromImages(trajRegion& t)
{
	std::vector<float> times = stack.getTimes();
	float endTime = times.back();
	float xv = (t.fx-t.ix)/endTime;
	float yv = (t.fy-t.iy)/endTime;
	for (int i=0; i<stack.imgCount(); ++i)
	{
		// Allow for fractional pixel coordinates
		float fractionalComp = std::pow(2.0, static_cast<float>(t.depth));
		float xp = fractionalComp*(t.ix + times[i] * xv); // +0.5;
		float yp = fractionalComp*(t.iy + times[i] * yv); // +0.5;
		int d = std::max(static_cast<int>(t.depth), 0);
		pooledPsi[i][d].maskObject(xp,yp, psf);
		pooledPhi[i][d].maskObject(xp,yp, psf);
	}
}

std::vector<RawImage>& KBMOSearch::getPsiImages() {
	return psiImages;
}

std::vector<RawImage>& KBMOSearch::getPhiImages() {
	return phiImages;
}

std::vector<std::vector<RawImage>>& KBMOSearch::getPsiPooled() {
	return pooledPsi;
}

std::vector<std::vector<RawImage>>& KBMOSearch::getPhiPooled() {
	return pooledPhi;
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
	if (start<0) throw std::runtime_error("start must be 0 or greater");
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

void KBMOSearch::startTimer(std::string message)
{
	if (debugInfo) {
		std::cout << message << "... " << std::flush;
		tStart = std::chrono::system_clock::now();
	}
}

void KBMOSearch::endTimer()
{
	if (debugInfo) {
		tEnd = std::chrono::system_clock::now();
		tDelta = tEnd-tStart;
		std::cout << " Took " << tDelta.count()
				<< " seconds.\n" << std::flush;
	}
}

} /* namespace kbmod */
