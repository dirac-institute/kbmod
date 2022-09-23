/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace kbmod {

KBMOSearch::KBMOSearch(ImageStack& imstack) : stack(imstack)
{
    totalPixelsRead = 0;
    regionsMaxed = 0;
    searchRegionsBounded = 0;
    individualEval = 0;
    nodesProcessed = 0;
    maxResultCount = 100000;
    debugInfo = false;
    psiPhiGenerated = false;

    // Default filtering arguments.
    percentiles = std::vector<float>{0.25, 0.5, 0.75};
    sigmaGCoeff = -1.0;
    minLH = -1.0;
    gpuFilter = false;

    // Set default values for the barycentric correction.
    baryCorrs = std::vector<baryCorrection>(stack.imgCount());
    useCorr = false;
}

void KBMOSearch::enableCorr(std::vector<float> pyBaryCorrCoeff)
{
    useCorr = true;
    for (int i=0; i<stack.imgCount(); i++){
        int j = i*6;
        baryCorrs[i].dx   = pyBaryCorrCoeff[j];
        baryCorrs[i].dxdx = pyBaryCorrCoeff[j+1];
        baryCorrs[i].dxdy = pyBaryCorrCoeff[j+2];
        baryCorrs[i].dy   = pyBaryCorrCoeff[j+3];
        baryCorrs[i].dydx = pyBaryCorrCoeff[j+4];
        baryCorrs[i].dydy = pyBaryCorrCoeff[j+5];
    } 
}
    
void KBMOSearch::enableGPUFilter(std::vector<float> pyPercentiles, 
                                 float pySigmaGCoeff,
                                 float pyMinLH)
{
    gpuFilter = true;
    percentiles = pyPercentiles;
    sigmaGCoeff = pySigmaGCoeff;
    minLH = pyMinLH;
}

void KBMOSearch::savePsiPhi(const std::string& path)
{
    preparePsiPhi();
    saveImages(path);
}

void KBMOSearch::search(int aSteps, int vSteps, float minAngle,
                        float maxAngle, float minVelocity, float maxVelocity, 
                        int minObservations)
{
    preparePsiPhi();
    createSearchList(aSteps, vSteps, minAngle, maxAngle, minVelocity, maxVelocity);

    startTimer("Creating interleaved psi/phi buffer");
    std::vector<float> interleavedPsiPhi;
    fillInterleavedPsiPhi(psiImages, phiImages, &interleavedPsiPhi);
    endTimer();

    results = std::vector<trajectory>(stack.getPPI()*RESULTS_PER_PIXEL);
    if (debugInfo) std::cout <<
            searchList.size() << " trajectories... \n" << std::flush;

    // Do the actual search on the GPU.
    startTimer("Searching");
    deviceSearchFilter(searchList.size(), stack.imgCount(), minObservations,
            interleavedPsiPhi.size(), stack.getPPI()*RESULTS_PER_PIXEL,
            searchList.data(), results.data(), stack.getTimesDataRef(),
            interleavedPsiPhi.data(), stack.getWidth(), stack.getHeight(),
            gpuFilter, &percentiles[0], sigmaGCoeff, minLH,
            useCorr, &baryCorrs[0]);
    endTimer();

    startTimer("Sorting results");
    sortResults();
    endTimer();
}

std::vector<trajRegion> KBMOSearch::regionSearch(
        float xVel, float yVel, float radius,
        float minLH, int minObservations)
{
    preparePsiPhi();
    startTimer("Searching regions");
    std::vector<trajRegion> res =
            resSearch(xVel, yVel, radius, minObservations, minLH);
    endTimer();
    if (debugInfo) {
        std::cout << totalPixelsRead <<
                " pixels read, computed bounds on "
                << regionsMaxed << " 2D regions for an average of "
                << static_cast<float>(totalPixelsRead)/static_cast<float>(regionsMaxed)
                << " pixels read per region\n"
                << searchRegionsBounded << " bounds computed on 4D regions\n"
                << individualEval << " individual trajectories LH computed\n";
    }
    return res;
}

void KBMOSearch::clearPsiPhi()
{
    psiPhiGenerated = false;
    psiImages = std::vector<RawImage>();
    phiImages = std::vector<RawImage>();
}

void KBMOSearch::preparePsiPhi()
{
    if (!psiPhiGenerated) {
        startTimer("Preparing psi and phi images");
        // Compute Phi and Psi from convolved images
        // while leaving masked pixels alone
        // Reinsert 0s for NO_DATA?
        clearPsiPhi();

        int num_images = stack.imgCount();
        int num_pixels = stack.getPPI();
        int w = stack.getWidth();
        int h = stack.getHeight();
        std::vector<float> currentPsi = std::vector<float>(num_pixels);
        std::vector<float> currentPhi = std::vector<float>(num_pixels);

        for (int i=0; i < num_images; ++i)
        {
            LayeredImage& img = stack.getSingleImage(i);
            float *sciArray = img.getSDataRef();
            float *varArray = img.getVDataRef();
            for (unsigned p=0; p < num_pixels; ++p)
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
            psiImages.push_back(RawImage(w, h, currentPsi));
            phiImages.push_back(RawImage(w, h, currentPhi));
        }
        endTimer();

        // Convolve psi and phi using the image specific PSFs.
        startTimer("Convolving images");
        for (int i = 0; i < stack.imgCount(); ++i)
        {
            LayeredImage& img = stack.getSingleImage(i);
            psiImages[i].convolve(img.getPSF());
            phiImages[i].convolve(img.getPSFSQ());
        }        
        endTimer();
        psiPhiGenerated = true;
    }
}

void KBMOSearch::repoolArea(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                            std::vector<PooledImage>& pooledPhi)
{
    // Repool small area of images after bright object
    // has been removed
    // This should probably be refactored in to multiple methods
    const std::vector<float>& times = stack.getTimes();
    const PointSpreadFunc& psf = stack.getSingleImage(0).getPSF();
    
    float xv = (t.fx-t.ix)/times.back();
    float yv = (t.fy-t.iy)/times.back();

    for (unsigned i = 0; i < pooledPsi.size(); ++i)
    {
        float x = t.ix + xv*times[i];
        float y = t.iy + yv*times[i];
        pooledPsi[i].repoolArea(x, y, psf.getDim());
        pooledPhi[i].repoolArea(x, y, psf.getDim());
    }
}

void KBMOSearch::saveImages(const std::string& path)
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

void KBMOSearch::fillInterleavedPsiPhi(
        const std::vector<RawImage>& psiImgs,
        const std::vector<RawImage>& phiImgs,
        std::vector<float>* interleaved)
{
    assert(interleaved != NULL);

    int num_images = psiImgs.size();
    assert(num_images > 0);
    assert(phiImgs.size() == num_images);

    int num_pixels = psiImgs[0].getPPI();
    assert(phiImgs[0].getPPI() == num_pixels);

    interleaved->clear();
    interleaved->reserve(2 * num_images * num_pixels);
    for (int i = 0; i < num_images; ++i)
    {
        const std::vector<float>& psiRef = psiImgs[i].getPixels();
        const std::vector<float>& phiRef = phiImgs[i].getPixels();
        for (unsigned p = 0; p < num_pixels; ++p)
        {
            interleaved->push_back(psiRef[p]);
            interleaved->push_back(phiRef[p]);
        }
    }
}

std::vector<trajRegion> KBMOSearch::resSearch(float xVel, float yVel,
        float radius, int minObservations, float minLH)
{
    startTimer("Pooling images");
    std::vector<PooledImage> pooledPsi = PoolMultipleImages(psiImages, POOL_MAX);
    std::vector<PooledImage> pooledPhi = PoolMultipleImages(phiImages, POOL_MIN);
    endTimer();

    int maxDepth = pooledPsi[0].numLevels() - 1;
    float finalTime = stack.getTimes().back();
    assert(maxDepth > 0 && maxDepth < 127);
    trajRegion root = {0.0, 0.0, 0.0, 0.0, static_cast<short>(maxDepth),
                       0, 0.0, 0.0};
    calculateLH(root, pooledPsi, pooledPhi);

    // Create a priority queue of potential trajectories.
    // with cmpLH = the function to sort trajectories.
    auto cmpLH = [](trajRegion a, trajRegion b)
            { return a.likelihood < b.likelihood; };
    std::priority_queue<trajRegion, std::vector<trajRegion>,
        decltype(cmpLH)> candidates(cmpLH);
    candidates.push(root);

    std::vector<trajRegion> fResults;
    while (!candidates.empty() && candidates.size() < 150000000)
    {
        nodesProcessed++;

        // Pop the top element of the priority queue.
        trajRegion t = candidates.top();
        candidates.pop();

        // Recalculate the likelihood in case it has changed due to
        // removing another trajectory. Filter the trajectory if it is
        // no longer good enough.
        calculateLH(t, pooledPsi, pooledPhi);
        if (t.likelihood < minLH || t.obs_count < minObservations)
            continue;

        // if the new score is lower, push it back into the queue
        if (t.likelihood<candidates.top().likelihood) {
            candidates.push(t);
            continue;
        }

        if (t.depth == 0) {
            // Remove the objects pixels from future searching
            // and make sure section of images are
            // repooled after object removal
            removeObjectFromImages(t, pooledPsi, pooledPhi);
            repoolArea(t, pooledPsi, pooledPhi);
            if (debugInfo) std::cout << "\nFound Candidate at x: " << t.ix << " y: " << t.iy << "\n";
            fResults.push_back(t);
            if (fResults.size() >= maxResultCount) break;
        } else {
            std::vector<trajRegion> sublist = subdivide(t);

            // Filter out subregions with invalid velocities.
            filterBounds(sublist, xVel, yVel, finalTime, radius);

            // Compute the likelihood of each of the subregions.
            for (auto& t : sublist) calculateLH(t, pooledPsi, pooledPhi);
            filterLH(sublist, minLH, minObservations);

            // Push the surviving subregions onto candidates.
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

std::vector<trajRegion>& KBMOSearch::filterBounds(std::vector<trajRegion>& tlist,
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

std::vector<trajRegion>& KBMOSearch::filterLH(
        std::vector<trajRegion>& tlist, float minLH, int minObs)
{
    tlist.erase(
            std::remove_if(tlist.begin(), tlist.end(),
                    std::bind([](trajRegion t, int mObs, float mLH) {
                        return t.obs_count<mObs || t.likelihood<mLH;
    }, std::placeholders::_1, minObs, minLH)),
    tlist.end());
    return tlist;
}

void KBMOSearch::calculateLH(trajRegion& t,
                             std::vector<PooledImage>& pooledPsi,
                             std::vector<PooledImage>& pooledPhi)
{

    const std::vector<float>& times = stack.getTimes();
    float endTime = times.back();
    float xv = (t.fx-t.ix)/endTime;
    float yv = (t.fy-t.iy)/endTime;
    // For region depths
    float fractionalComp = std::pow(2.0, static_cast<float>(t.depth));
    int d = std::max(static_cast<int>(t.depth), 0);
    int size = 1 << static_cast<int>(t.depth);

    float psiSum = 0.0;
    float phiSum = 0.0;
    t.obs_count = 0;

    // Second pass removes outliers
    for (int i=0; i<stack.imgCount(); ++i)
    {
        float tempPsi = 0.0;
        float tempPhi = 0.0;
        // Read from region rather than single pixel
        if (t.depth > 0) {
            searchRegionsBounded++;
            float x = t.ix+0.5 + times[i] * xv;
            float y = t.iy+0.5 + times[i] * yv;
            int size = 1 << static_cast<int>(t.depth);
            tempPsi = findExtremeInRegion(x, y, size, pooledPsi[i], POOL_MAX);
            if (tempPsi == NO_DATA) continue;
            tempPhi = findExtremeInRegion(x, y, size, pooledPhi[i], POOL_MIN);
        } else {
            individualEval++;
            // Use exact pixels to be consistent with later filtering.
            int xp = t.ix + int(times[i] * xv + 0.5);
            int yp = t.iy + int(times[i] * yv + 0.5);
            tempPsi = pooledPsi[i].getImage(0).getPixel(xp, yp);
            if (tempPsi == NO_DATA) continue;
            tempPhi = pooledPhi[i].getImage(0).getPixel(xp, yp);
        }
        psiSum += tempPsi;
        phiSum += tempPhi;
        t.obs_count++;
    }

    t.likelihood = phiSum > 0.0 ? psiSum/sqrt(phiSum) : NO_DATA;
    t.flux = phiSum > 0.0 ? psiSum/phiSum : NO_DATA;
}

float KBMOSearch::findExtremeInRegion(float x, float y,
    int size, PooledImage& pooledImgs, int poolType)
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
            pooledImgs.getImage(depth).extremeInRegion(lx, ly, hx-1, hy-1, poolType);
    return regionExtreme;
}

void KBMOSearch::removeObjectFromImages(trajRegion& t,
                                        std::vector<PooledImage>& pooledPsi,
                                        std::vector<PooledImage>& pooledPhi)
{
    const std::vector<float>& times = stack.getTimes();
    float endTime = times.back();
    float xv = (t.fx - t.ix)/endTime;
    float yv = (t.fy - t.iy)/endTime;
    for (int i = 0; i < stack.imgCount(); ++i)
    {
        const PointSpreadFunc& psf = stack.getSingleImage(i).getPSF();

        // Allow for fractional pixel coordinates
        float fractionalComp = std::pow(2.0, static_cast<float>(t.depth));
        float xp = fractionalComp*(t.ix + times[i] * xv); // +0.5;
        float yp = fractionalComp*(t.iy + times[i] * yv); // +0.5;
        int d = std::max(static_cast<int>(t.depth), 0);
        pooledPsi[i].getImage(d).maskObject(xp, yp, psf);
        pooledPhi[i].getImage(d).maskObject(xp, yp, psf);
    }
}

trajectory KBMOSearch::convertTraj(trajRegion& t)
{
    const std::vector<float>& times = stack.getTimes();
    float endTime = times.back();
    float xv = (t.fx-t.ix)/endTime;
    float yv = (t.fy-t.iy)/endTime;
    trajectory tb;
    tb.lh = t.likelihood;
    tb.flux = t.flux;
    tb.obsCount = t.obs_count;
    tb.x = t.ix;
    tb.y = t.iy;
    tb.xVel = xv;
    tb.yVel = yv;
    return tb;
}

std::vector<RawImage> KBMOSearch::medianStamps(const std::vector<trajectory>& t_array,
                                               const std::vector<std::vector<int>>& goodIdx,
                                               int radius)
{
    int numResults = t_array.size();
    int dim = radius*2+1;

    std::vector<LayeredImage>& imgs = stack.getImages();
    size_t N = imgs.size() / 2;
    std::vector<RawImage> results(numResults);
    omp_set_num_threads(16);

    #pragma omp parallel for
    for (int s = 0; s < numResults; ++s)
    {
        // Create stamps around the current trajectory.
        std::vector<RawImage> stamps;
        trajectory t = t_array[s];
        for (int i = 0; i < goodIdx[s].size(); ++i)
        {
            if (goodIdx[s][i] == 1)
            {
                std::array<float,2> pos = getTrajPos(t, i);
                stamps.push_back(imgs[i].getScience().createStamp(pos[0], pos[1], radius, false, true));
            }
        }

        // Compute the median of those stamps.
        results[s] = createMedianImage(stamps);
    }
    omp_set_num_threads(1);

    return(results);
}

std::vector<RawImage> KBMOSearch::meanStamps(const std::vector<trajectory>& t_array,
                                             const std::vector<std::vector<int>>& goodIdx,
                                             int radius)
{
    int numResults = t_array.size();
    int dim = radius*2+1;

    std::vector<LayeredImage>& imgs = stack.getImages();
    size_t N = imgs.size() / 2;
    std::vector<RawImage> results(numResults);
    omp_set_num_threads(16);

    #pragma omp parallel for
    for (int s = 0; s < numResults; ++s)
    {
        // Create stamps around the current trajectory.
        std::vector<RawImage> stamps;
        trajectory t = t_array[s];
        for (int i = 0; i < goodIdx[s].size(); ++i)
        {
            if (goodIdx[s][i] == 1)
            {
                std::array<float,2> pos = getTrajPos(t, i);
                stamps.push_back(imgs[i].getScience().createStamp(pos[0], pos[1], radius, false, true));
            }
        }

        // Compute the mean of those stamps.
        results[s] = createMeanImage(stamps);
    }
    omp_set_num_threads(1);

    return(results);
}

std::vector<RawImage> KBMOSearch::createStamps(trajectory t, int radius,
                                               const std::vector<RawImage*>& imgs,
                                               bool interpolate)
{
    if (radius<0) throw std::runtime_error("stamp radius must be at least 0");
    std::vector<RawImage> stamps;
    for (int i=0; i < imgs.size(); ++i)
    {
        std::array<float,2> pos = getTrajPos(t, i);
        stamps.push_back(imgs[i]->createStamp(pos[0], pos[1], radius, interpolate, false));
    }
    return stamps;
}

std::array<float,2> KBMOSearch::getTrajPos(trajectory t, int i){
    float time = stack.getTimes()[i];

    std::array<float,2> pos;

    pos[0] = t.x + time * t.xVel;
    pos[1] = t.y + time * t.yVel;

    if (useCorr) {
        baryCorrection bc = baryCorrs[i];
        pos[0] += bc.dx + t.x*bc.dxdx + t.y*bc.dxdy;
        pos[1] += bc.dy + t.x*bc.dydx + t.y*bc.dydy;
    }

    return pos;
}

std::vector<float> KBMOSearch::createCurves(trajectory t, std::vector<RawImage*> imgs)
{
    /*Create a lightcurve from an image along a trajectory
     *
     *  INPUT-
     *    trajectory t - The trajectory along which to compute the lightcurve
     *    std::vector<RawImage*> imgs - The image from which to compute the
     *      trajectory. Most likely a psiImage or a phiImage.
     *  Output-
     *    std::vector<float> lightcurve - The computed trajectory
     */

    int imgSize = imgs.size();
    std::vector<float> lightcurve;
    lightcurve.reserve(imgSize);
    const std::vector<float>& times = stack.getTimes();
    for (int i=0; i<imgSize; ++i)
    {
        /* Do not use getPixelInterp(), because results from createCurves must
         * be able to recover the same likelihoods as the ones reported by the
         * gpu search.*/
        float pixVal;
        if (useCorr){
            std::array<float,2> pos = getTrajPos(t, i);
            pixVal = imgs[i]->getPixel(
                int(pos[0] + 0.5),
                int(pos[1] + 0.5));
        }
        /* Does not use getTrajPos to be backwards compatible with Hits_Rerun */
        else {
            pixVal = imgs[i]->getPixel(
                t.x + int(times[i] * t.xVel + 0.5),
                t.y + int(times[i] * t.yVel + 0.5));
        }
        if (pixVal == NO_DATA) pixVal = 0.0;
        lightcurve.push_back(pixVal);
    }
    return lightcurve;
}

std::vector<RawImage> KBMOSearch::psiStamps(trajRegion& t, int radius)
{
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : psiImages) imgs.push_back(&im);
    return createStamps(convertTraj(t), radius, imgs, true);
}

std::vector<RawImage> KBMOSearch::phiStamps(trajRegion& t, int radius)
{
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : phiImages) imgs.push_back(&im);
    return createStamps(convertTraj(t), radius, imgs, true);
}

std::vector<RawImage> KBMOSearch::scienceStamps(trajectory& t, int radius)
{
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());
    return createStamps(t, radius, imgs, true);
}

std::vector<RawImage> KBMOSearch::scienceStamps(trajRegion& t, int radius)
{
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());
    return createStamps(convertTraj(t), radius, imgs, true);
}

RawImage KBMOSearch::stackedScience(trajectory& t, int radius)
{
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());

    std::vector<RawImage> stamps = createStamps(t, radius, imgs, false);
    RawImage summedStamp = createSummedImage(stamps);
    return summedStamp;
}

RawImage KBMOSearch::stackedScience(trajRegion& t, int radius)
{
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());

    std::vector<RawImage> stamps = createStamps(convertTraj(t), radius, imgs, false);
    RawImage summedStamp = createSummedImage(stamps);
    return summedStamp;
}

std::vector<RawImage> KBMOSearch::summedScience(const std::vector<trajectory>& t_array,
                                                int radius)
{
    int numResults = t_array.size();
    std::vector<RawImage> results(numResults);

    // Extract the science images.
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());

    // Build the result for each trajectory.
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int s = 0; s < numResults; ++s)
    {
        // Create stamps around the current trajectory.
        trajectory t = t_array[s];
        std::vector<RawImage> stamps = createStamps(t, radius, imgs, false);

        // Compute the summation of those stamps.
        results[s] = createSummedImage(stamps);
    }
    omp_set_num_threads(1);

    return(results);
}

std::vector<RawImage> KBMOSearch::psiStamps(trajectory& t, int radius)
{
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : psiImages) imgs.push_back(&im);
    return createStamps(t, radius, imgs, true);
}

std::vector<RawImage> KBMOSearch::phiStamps(trajectory& t, int radius)
{
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : phiImages) imgs.push_back(&im);
    return createStamps(t, radius, imgs, true);
}

std::vector<float> KBMOSearch::psiCurves(trajectory& t)
{
    /*Generate a psi lightcurve for further analysis
     *  INPUT-
     *    trajectory& t - The trajectory along which to find the lightcurve
     *  OUTPUT-
     *    std::vector<float> - A vector of the lightcurve values
     */
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : psiImages) imgs.push_back(&im);
    return createCurves(t, imgs);
}

std::vector<float> KBMOSearch::phiCurves(trajectory& t)
{
    /*Generate a phi lightcurve for further analysis
     *  INPUT-
     *    trajectory& t - The trajectory along which to find the lightcurve
     *  OUTPUT-
     *    std::vector<float> - A vector of the lightcurve values
     */
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : phiImages) imgs.push_back(&im);
    return createCurves(t, imgs);
}

std::vector<RawImage>& KBMOSearch::getPsiImages() {
    return psiImages;
}

std::vector<RawImage>& KBMOSearch::getPhiImages() {
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

void KBMOSearch::filterResultsLH(float minLH)
{
    results.erase(
            std::remove_if(results.begin(), results.end(),
                    std::bind([](trajectory t, float cutoff) {
                        return t.lh<cutoff;
    }, std::placeholders::_1, minLH)),
    results.end());
}

std::vector<trajectory> KBMOSearch::getResults(int start, int count){
    if (start<0) throw std::runtime_error("start must be 0 or greater");
    return std::vector<trajectory>(results.begin()+start, results.begin()+start+count);
}

void KBMOSearch::saveResults(const std::string& path, float portion)
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

void KBMOSearch::startTimer(const std::string& message)
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
