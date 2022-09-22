/*
 * KBMOSearch.h
 *
 * Created on: Jun 28, 2017
 * Author: kbmod-usr
 *
 * The KBMOSearch class holds all of the information and functions
 * to perform the core stacked search.
 */

#ifndef KBMODSEARCH_H_
#define KBMODSEARCH_H_

#include <parallel/algorithm>
#include <algorithm>
#include <functional>
#include <queue>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <assert.h>
#include <float.h>
#include "common.h"
#include "PointSpreadFunc.h"
#include "ImageStack.h"
#include "PooledImage.h"

namespace kbmod {

extern "C" void
deviceSearchFilter(
        int trajCount, int imageCount, int minObservations, int psiPhiSize,
        int resultsCount, trajectory *trajectoriesToSearch, trajectory *bestTrajects,
        float *imageTimes, float *interleavedPsiPhi, int width, int height,
        bool doFilter, float percentiles[2], float sigmaGCoeff,
        float minLH, bool useCorr, baryCorrection *baryCorrs);

class KBMOSearch {
public:
    KBMOSearch(ImageStack& imstack);

    int numImages() const { return stack.imgCount(); }
    const ImageStack& getImageStack() const { return stack; }

    void setDebug(bool d) { debugInfo = d; };

    // The primary search functions.
    void enableGPUFilter(std::vector<float> pyPercentiles, 
                         float pySigmaGCoeff, float pyMinLH);
    void enableCorr(std::vector<float> pyBaryCorrCoeff);
    void search(int aSteps, int vSteps, float minAngle,
                float maxAngle, float minVelocity, float maxVelocity, 
                int minObservations);
    std::vector<trajRegion> regionSearch(float xVel, float yVel,
            float radius, float minLH, int minObservations);

    // Gets the vector of result trajectories.
    std::vector<trajectory> getResults(int start, int end);

    // Get the predicted (pixel) positions for a given trajectory.
    pixelPos getTrajPos(trajectory t, int i) const;
    std::vector<pixelPos> getTrajPositions(trajectory& t) const;

    // Filters the results based on various parameters.
    void filterResults(int minObservations);
    void filterResultsLH(float minLH);
    std::vector<trajRegion>& filterLH(std::vector<trajRegion>& tlist, float minLH, int minObs);
    std::vector<trajRegion>& filterBounds(std::vector<trajRegion>& tlist,
            float xVel, float yVel, float ft, float radius);

    // Compute the likelihood of trajRegion results.
    void calculateLH(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                     std::vector<PooledImage>& pooledPhi);

    float squareSDF(float scale, float centerX, float centerY,
            float pointX, float pointY);
    float findExtremeInRegion(float x, float y, int size,
            PooledImage& pooledImg, int poolType);

    // Converts a trajRegion result into a trajectory result.
    trajectory convertTraj(trajRegion& t);

    // Subdivides a trajRegion into 16 subregions.
    std::vector<trajRegion> subdivide(trajRegion& t);

    // Functions to create and access stamps around proposed trajectories or
    // regions. Used to visualize the results.
    // These functions drop pixels with NO_DATA from the computation.
    std::vector<RawImage> medianStamps(const std::vector<trajectory>& t_array,
                                       const std::vector<std::vector<int>>& goodIdx,
                                       int radius);
    std::vector<RawImage> meanStamps(const std::vector<trajectory>& t_array,
                                     const std::vector<std::vector<int>>& goodIdx,
                                     int radius);

    // Creates science stamps (or a summed stamp) around a
    // trajectory, trajRegion, or vector of trajectories.
    // These functions replace NO_DATA with a value of 0.0.
    std::vector<RawImage> scienceStamps(trajRegion& t, int radius);
    std::vector<RawImage> scienceStamps(trajectory& t, int radius);
    RawImage stackedScience(trajRegion& t, int radius);
    RawImage stackedScience(trajectory& t, int radius);
    std::vector<RawImage> summedScience(const std::vector<trajectory>& t_array,
                                        int radius);

    // Getters for the Psi and Phi data, including pooled
    // and stamped versions.
    std::vector<RawImage>& getPsiImages();
    std::vector<RawImage>& getPhiImages();
    std::vector<RawImage> psiStamps(trajectory& t, int radius);
    std::vector<RawImage> phiStamps(trajectory& t, int radius);
    std::vector<RawImage> psiStamps(trajRegion& t, int radius);
    std::vector<RawImage> phiStamps(trajRegion& t, int radius);
    std::vector<float> psiCurves(trajectory& t);
    std::vector<float> phiCurves(trajectory& t);

    // Save results or internal data products to a file.
    void saveResults(const std::string& path, float fraction);
    void savePsiPhi(const std::string& path);

    // Helper functions for computing Psi and Phi.
    void preparePsiPhi();
    void clearPsiPhi();

    virtual ~KBMOSearch() {};

private:
    std::vector<trajRegion> resSearch(float xVel, float yVel,
            float radius, int minObservations, float minLH);
    void removeObjectFromImages(trajRegion& t,
                                std::vector<PooledImage>& pooledPsi,
                                std::vector<PooledImage>& pooledPhi);
    void saveImages(const std::string& path);
    void sortResults();
    std::vector<float> createCurves(trajectory t, std::vector<RawImage*> imgs);

    // Fill an interleaved vector for the GPU functions.
    void fillInterleavedPsiPhi(const std::vector<RawImage>& psiImgs,
                               const std::vector<RawImage>& phiImgs,
                               std::vector<float>* interleaved);

    // Functions to create and access stamps around proposed trajectories or
    // regions. Used to visualize the results.
    // This function replaces NO_DATA with a value of 0.0.
    std::vector<RawImage> createStamps(trajectory t, int radius,
                                       const std::vector<RawImage*>& imgs,
                                       bool interpolate);

    // Creates list of trajectories to search.
    void createSearchList(int angleSteps, int veloctiySteps, float minAngle,
                          float maxAngle, float minVelocity, float maxVelocity);

    // Helper functions for the pooled data.
    void repoolArea(trajRegion& t, std::vector<PooledImage>& pooledPsi,
                    std::vector<PooledImage>& pooledPhi);

    // Helper functions for timing operations of the search.
    void startTimer(const std::string& message);
    void endTimer();

    long int totalPixelsRead;
    long int regionsMaxed;
    long int searchRegionsBounded;
    long int individualEval;
    long long nodesProcessed;
    unsigned maxResultCount;
    bool psiPhiGenerated;
    bool debugInfo;
    ImageStack stack;
    std::vector<trajectory> searchList;
    std::vector<RawImage> psiImages;
    std::vector<RawImage> phiImages;
    std::vector<trajectory> results;

    // Variables for the timer.
    std::chrono::time_point<std::chrono::system_clock> tStart, tEnd;
    std::chrono::duration<double> tDelta;

    // Parameters for on GPU filtering.
    bool gpuFilter;
    float sigmaGCoeff;
    float minLH;
    std::vector<float> percentiles;

    // Parameters to do barycentric corrections.
    bool useCorr;
    std::vector<baryCorrection> baryCorrs;
};

} /* namespace kbmod */

#endif /* KBMODSEARCH_H_ */
