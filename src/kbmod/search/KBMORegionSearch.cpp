/*
 * KBMORegionSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMORegionSearch.h"

namespace search {

KBMORegionSearch::KBMORegionSearch(ImageStack& imstack) : KBMOSearch(imstack) {
    totalPixelsRead = 0;
    regionsMaxed = 0;
    searchRegionsBounded = 0;
    individualEval = 0;
    nodesProcessed = 0;
}

std::vector<trajRegion> KBMORegionSearch::regionSearch(float xVel, float yVel, float radius, float minLH,
                                                       int minObservations) {
    preparePsiPhi();
    startTimer("Searching regions");
    std::vector<trajRegion> res = resSearch(xVel, yVel, radius, minObservations, minLH);
    endTimer();
    if (debugInfo) {
        std::cout << totalPixelsRead << " pixels read, computed bounds on " << regionsMaxed
                  << " 2D regions for an average of "
                  << static_cast<float>(totalPixelsRead) / static_cast<float>(regionsMaxed)
                  << " pixels read per region\n"
                  << searchRegionsBounded << " bounds computed on 4D regions\n"
                  << individualEval << " individual trajectories LH computed\n";
    }
    return res;
}

void KBMORegionSearch::repoolArea(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                                  std::vector<PooledImage>& pooledPhi) {
    // Repool small area of images after bright object
    // has been removed
    // This should probably be refactored in to multiple methods
    const std::vector<float>& times = stack.getTimes();
    const PointSpreadFunc& psf = stack.getSingleImage(0).getPSF();

    float xv = (t.fx - t.ix) / times.back();
    float yv = (t.fy - t.iy) / times.back();

    for (unsigned i = 0; i < pooledPsi.size(); ++i) {
        float x = t.ix + xv * times[i];
        float y = t.iy + yv * times[i];
        pooledPsi[i].repoolArea(x, y, psf.getDim());
        pooledPhi[i].repoolArea(x, y, psf.getDim());
    }
}

std::vector<trajRegion> KBMORegionSearch::resSearch(float xVel, float yVel, float radius, int minObservations,
                                                    float minLH) {
    startTimer("Pooling images");
    std::vector<PooledImage> pooledPsi = PoolMultipleImages(psiImages, POOL_MAX, false);
    std::vector<PooledImage> pooledPhi = PoolMultipleImages(phiImages, POOL_MIN, false);
    endTimer();

    int maxDepth = pooledPsi[0].numLevels() - 1;
    float finalTime = stack.getTimes().back();
    assert(maxDepth > 0 && maxDepth < 127);
    trajRegion root = {0.0, 0.0, 0.0, 0.0, static_cast<short>(maxDepth), 0, 0.0, 0.0};
    calculateLH(root, pooledPsi, pooledPhi);

    // Create a priority queue of potential trajectories.
    // with cmpLH = the function to sort trajectories.
    auto cmpLH = [](trajRegion a, trajRegion b) { return a.likelihood < b.likelihood; };
    std::priority_queue<trajRegion, std::vector<trajRegion>, decltype(cmpLH)> candidates(cmpLH);
    candidates.push(root);

    std::vector<trajRegion> fResults;
    while (!candidates.empty() && candidates.size() < 150000000) {
        nodesProcessed++;

        // Pop the top element of the priority queue.
        trajRegion t = candidates.top();
        candidates.pop();

        // Recalculate the likelihood in case it has changed due to
        // removing another trajectory. Filter the trajectory if it is
        // no longer good enough.
        calculateLH(t, pooledPsi, pooledPhi);
        if (t.likelihood < minLH || t.obs_count < minObservations) continue;

        // if the new score is lower, push it back into the queue
        if (t.likelihood < candidates.top().likelihood) {
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
            std::vector<trajRegion> sublist = subdivideTrajRegion(t);

            // Filter out subregions with invalid velocities.
            filterBounds(sublist, xVel, yVel, finalTime, radius);

            // Compute the likelihood of each of the subregions.
            for (auto& t : sublist) calculateLH(t, pooledPsi, pooledPhi);
            filterTrajRegionsLH(sublist, minLH, minObservations);

            // Push the surviving subregions onto candidates.
            for (auto& nt : sublist) candidates.push(nt);
        }
    }
    std::cout << std::endl;
    return fResults;
}

std::vector<trajRegion>& KBMORegionSearch::filterBounds(std::vector<trajRegion>& tlist, float xVel,
                                                        float yVel, float ft, float radius) {
    tlist.erase(
            std::remove_if(
                    tlist.begin(), tlist.end(),
                    std::bind(
                            [](trajRegion t, KBMORegionSearch* s, float xv, float yv, float finalT,
                               float rad) {
                                // 2 raised to the depth power
                                float scale = std::pow(2.0, static_cast<float>(t.depth));
                                float centerX = scale * (t.fx + 0.5);
                                float centerY = scale * (t.fy + 0.5);
                                float posX = scale * (t.ix + 0.5) + xv * finalT;
                                float posY = scale * (t.iy + 0.5) + yv * finalT;
                                // 2D box signed distance function
                                float dist = s->squareSDF(scale, centerX, centerY, posX - 0.5 * scale,
                                                          posY + 0.5 * scale);
                                dist = std::min(dist, s->squareSDF(scale, centerX, centerY,
                                                                   posX + 0.5 * scale, posY + 0.5 * scale));
                                dist = std::min(dist, s->squareSDF(scale, centerX, centerY,
                                                                   posX - 0.5 * scale, posY - 0.5 * scale));
                                dist = std::min(dist, s->squareSDF(scale, centerX, centerY,
                                                                   posX + 0.5 * scale, posY - 0.5 * scale));
                                return (dist - rad) > 0.0;
                            },
                            std::placeholders::_1, this, xVel, yVel, ft, radius)),
            tlist.end());
    return tlist;
}

float KBMORegionSearch::squareSDF(float scale, float centerX, float centerY, float pointX, float pointY) {
    float dx = pointX - centerX;
    float dy = pointY - centerY;
    float xn = std::abs(dx) - scale * 0.5f;
    float yn = std::abs(dy) - scale * 0.5f;
    float xk = std::min(xn, 0.0f);
    float yk = std::min(yn, 0.0f);
    float xm = std::max(xn, 0.0f);
    float ym = std::max(yn, 0.0f);
    return sqrt(xm * xm + ym * ym) + std::max(xk, yk);
}

void KBMORegionSearch::calculateLH(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                                   std::vector<PooledImage>& pooledPhi) {
    const std::vector<float>& times = stack.getTimes();
    float endTime = times.back();
    float xv = (t.fx - t.ix) / endTime;
    float yv = (t.fy - t.iy) / endTime;
    // For region depths
    float fractionalComp = std::pow(2.0, static_cast<float>(t.depth));
    int d = std::max(static_cast<int>(t.depth), 0);
    int size = 1 << static_cast<int>(t.depth);

    float psiSum = 0.0;
    float phiSum = 0.0;
    t.obs_count = 0;

    // Second pass removes outliers
    for (int i = 0; i < stack.imgCount(); ++i) {
        float tempPsi = 0.0;
        float tempPhi = 0.0;
        // Read from region rather than single pixel
        if (t.depth > 0) {
            searchRegionsBounded++;
            float x = t.ix + 0.5 + times[i] * xv;
            float y = t.iy + 0.5 + times[i] * yv;
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

    t.likelihood = phiSum > 0.0 ? psiSum / sqrt(phiSum) : NO_DATA;
    t.flux = phiSum > 0.0 ? psiSum / phiSum : NO_DATA;
}

float KBMORegionSearch::findExtremeInRegion(float x, float y, int size, PooledImage& pooledImgs,
                                            int poolType) {
    regionsMaxed++;
    // check that maxSize is a power of two
    assert((size & (-size)) == size);
    x *= static_cast<float>(size);
    y *= static_cast<float>(size);
    int sizeToRead = std::max(size / REGION_RESOLUTION, 1);
    int depth = 0;
    // computer integer log2
    int tempLog = sizeToRead;
    while (tempLog >>= 1) ++depth;
    float s = static_cast<float>(size) * 0.5;
    // lower left corner of region
    int lx = static_cast<int>(floor(x - s));
    int ly = static_cast<int>(floor(y - s));
    // Round lower corner down to align larger pixel size
    lx = lx / sizeToRead;
    ly = ly / sizeToRead;
    // upper right corner of region
    int hx = static_cast<int>(ceil(x + s));
    int hy = static_cast<int>(ceil(y + s));
    // Round Upper corner up to align larger pixel size
    hx = (hx + sizeToRead - 1) / sizeToRead;
    hy = (hy + sizeToRead - 1) / sizeToRead;
    float regionExtreme = pooledImgs.getImage(depth).extremeInRegion(lx, ly, hx - 1, hy - 1, poolType);
    return regionExtreme;
}

void KBMORegionSearch::removeObjectFromImages(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                                              std::vector<PooledImage>& pooledPhi) {
    const std::vector<float>& times = stack.getTimes();
    float endTime = times.back();
    float xv = (t.fx - t.ix) / endTime;
    float yv = (t.fy - t.iy) / endTime;
    for (int i = 0; i < stack.imgCount(); ++i) {
        const PointSpreadFunc& psf = stack.getSingleImage(i).getPSF();

        // Allow for fractional pixel coordinates
        float fractionalComp = std::pow(2.0, static_cast<float>(t.depth));
        float xp = fractionalComp * (t.ix + times[i] * xv);  // +0.5;
        float yp = fractionalComp * (t.iy + times[i] * yv);  // +0.5;
        int d = std::max(static_cast<int>(t.depth), 0);
        pooledPsi[i].getImage(d).maskObject(xp, yp, psf);
        pooledPhi[i].getImage(d).maskObject(xp, yp, psf);
    }
}

std::vector<RawImage> KBMORegionSearch::psiStamps(trajRegion& t, int radius) {
    preparePsiPhi();
    const float endTime = stack.getTimes().back();

    std::vector<RawImage*> imgs;
    for (auto& im : psiImages) imgs.push_back(&im);
    return createStamps(convertTrajRegion(t, endTime), radius, imgs, true);
}

std::vector<RawImage> KBMORegionSearch::phiStamps(trajRegion& t, int radius) {
    preparePsiPhi();
    const float endTime = stack.getTimes().back();

    std::vector<RawImage*> imgs;
    for (auto& im : phiImages) imgs.push_back(&im);
    return createStamps(convertTrajRegion(t, endTime), radius, imgs, true);
}

std::vector<RawImage> KBMORegionSearch::scienceStamps(trajRegion& t, int radius) {
    const float endTime = stack.getTimes().back();
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());
    return createStamps(convertTrajRegion(t, endTime), radius, imgs, true);
}

RawImage KBMORegionSearch::stackedScience(trajRegion& t, int radius) {
    std::vector<RawImage*> imgs;
    for (auto& im : stack.getImages()) imgs.push_back(&im.getScience());

    const float endTime = stack.getTimes().back();
    std::vector<RawImage> stamps = createStamps(convertTrajRegion(t, endTime), radius, imgs, false);
    RawImage summedStamp = createSummedImage(stamps);
    return summedStamp;
}

} /* namespace search */
