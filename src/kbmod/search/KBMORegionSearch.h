/*
 * KBMORegionSearch.h
 *
 * Created on: Jun 28, 2017
 * Author: kbmod-usr
 *
 * A derived class from KBMOSearch that handles region search specific functions.
 */

#ifndef KBMODREGIONSEARCH_H_
#define KBMODREGIONSEARCH_H_

#include "common.h"
#include "ImageStack.h"
#include "KBMOSearch.h"
#include "PooledImage.h"
#include "RawImage.h"
#include <queue>
#include <vector>

namespace search {

class KBMORegionSearch : public KBMOSearch {
public:
    KBMORegionSearch(ImageStack& imstack);

    std::vector<trajRegion> regionSearch(float xVel, float yVel, float radius, float minLH,
                                         int minObservations);

    // Filters the results based on various parameters.
    std::vector<trajRegion>& filterLH(std::vector<trajRegion>& tlist, float minLH, int minObs);
    std::vector<trajRegion>& filterBounds(std::vector<trajRegion>& tlist, float xVel, float yVel, float ft,
                                          float radius);

    // Compute the likelihood of trajRegion results.
    void calculateLH(trajRegion& t, std::vector<PooledImage>& pooledPsi, std::vector<PooledImage>& pooledPhi);

    float squareSDF(float scale, float centerX, float centerY, float pointX, float pointY);
    float findExtremeInRegion(float x, float y, int size, PooledImage& pooledImg, int poolType);

    // Creates science stamps (or a summed stamp) around a
    // trajectory, trajRegion, or vector of trajectories.
    // These functions replace NO_DATA with a value of 0.0.
    std::vector<RawImage> scienceStamps(trajRegion& t, int radius);
    RawImage stackedScience(trajRegion& t, int radius);

    // Getters for the Psi and Phi data, including pooled
    // and stamped versions.
    std::vector<RawImage> psiStamps(trajRegion& t, int radius);
    std::vector<RawImage> phiStamps(trajRegion& t, int radius);

    virtual ~KBMORegionSearch(){};

private:
    std::vector<trajRegion> resSearch(float xVel, float yVel, float radius, int minObservations, float minLH);
    void removeObjectFromImages(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                                std::vector<PooledImage>& pooledPhi);

    // Helper functions for the pooled data.
    void repoolArea(trajRegion& t, std::vector<PooledImage>& pooledPsi, std::vector<PooledImage>& pooledPhi);

    long int totalPixelsRead;
    long int regionsMaxed;
    long int searchRegionsBounded;
    long int individualEval;
    long long nodesProcessed;
};

} /* namespace search */

#endif /* KBMODREGIONSEARCH_H_ */
