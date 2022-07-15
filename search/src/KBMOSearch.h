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
deviceSearch(
        int trajCount, int imageCount, int minObservations, int psiPhiSize,
        int resultsCount, trajectory *trajectoriesToSearch, trajectory *bestTrajects,
        float *imageTimes, float *interleavedPsiPhi, int width, int height);

extern "C" void
deviceSearchFilter(
        int trajCount, int imageCount, int minObservations, int psiPhiSize,
        int resultsCount, trajectory *trajectoriesToSearch, trajectory *bestTrajects,
        float *imageTimes, float *interleavedPsiPhi, int width, int height,
        float percentiles[2], float sigmaGCoeff, float centralMomLims[5],
        float minLH);

extern "C" void
devicePooledSetup(int imageCount, int depth, float *times, int *dimensions, float *interleavedImages,
        float **deviceTimes, float **deviceImages, int **deviceDimensions); // Dimensions?

extern "C" void
devicePooledTeardown(float **deviceTimes, float **deviceImages, int **dimensions);

extern "C" void
deviceLHBatch(int imageCount, int depth, int regionCount, trajRegion *regions,
        float **deviceTimes, float **deviceImages, float **deviceDimensions);

class KBMOSearch {
public:
    KBMOSearch(ImageStack& imstack, PointSpreadFunc& PSF);

    void setDebug(bool d) { debugInfo = d; };

    // The primary search functions.
    void gpu(int aSteps, int vSteps, float minAngle, float maxAngle,
            float minVelocity, float maxVelocity, int minObservations);
    void gpuFilter(int aSteps, int vSteps, float minAngle, float maxAngle,
            float minVelocity, float maxVelocity, int minObservations,
            std::vector<float> pyPercentiles, float pySigmaGCoeff,
            std::vector<float> pyCentralMomLims, float minLH);
    void cpu(int aSteps, int vSteps, float minAngle, float maxAngle,
            float minVelocity, float maxVelocity, int minObservations);
    std::vector<trajRegion> regionSearch(float xVel, float yVel,
            float radius, float minLH, int minObservations);

    // Gets the vector of result trajectories.
    std::vector<trajectory> getResults(int start, int end);

    // Filters the results based on various parameters.
    void filterResults(int minObservations);
    void filterResultsLH(float minLH);
    std::vector<trajRegion>& filterLH(std::vector<trajRegion>& tlist, float minLH, int minObs);
    std::vector<trajRegion>& filterBounds(std::vector<trajRegion>& tlist,
            float xVel, float yVel, float ft, float radius);

    // Compute the likelihood of trajRegion results.
    trajRegion& calculateLH(trajRegion& t);
    std::vector<trajRegion>& calculateLHBatch(std::vector<trajRegion>& tlist);

    int biggestFit(int x, int y, int maxX, int maxY); // inline?
    float squareSDF(float scale, float centerX, float centerY,
            float pointX, float pointY);
    float pixelExtreme(float pixel, float prev, int poolType);
    float findExtremeInRegion(float x, float y, int size, 
            PooledImage& pooledImg, int poolType);

    // Converts a trajRegion result into a trajectory result.
    trajectory convertTraj(trajRegion& t);

    // Subdivides a trajRegion into 16 subregions.
    std::vector<trajRegion> subdivide(trajRegion& t);

    // Functions to create and access stamps around proposed trajectories or
    // regions. Used to visualize the results.
    std::vector<RawImage> createStamps(trajectory t, int radius, std::vector<RawImage*> imgs);
    std::vector<RawImage> medianStamps(std::vector<trajectory> t_array,
                                       std::vector<std::vector<int>> goodIdx,
                                       int radius);
    std::vector<RawImage> createMedianBatch(int radius,  std::vector<RawImage*> imgs);
    std::vector<RawImage> summedStamps(std::vector<trajectory> t_array, int radius);
    RawImage stackedStamps(trajectory t, int radius, std::vector<RawImage*> imgs);
    std::vector<RawImage> scienceStamps(trajRegion& t, int radius);
    std::vector<RawImage> scienceStamps(trajectory& t, int radius);

    // Creates stack science stamps around a trajectory or trajRegion.
    RawImage stackedScience(trajRegion& t, int radius);
    RawImage stackedScience(trajectory& t, int radius);

    // Getters for the Psi and Phi data, including pooled
    // and stamped versions.
    std::vector<RawImage>& getPsiImages();
    std::vector<RawImage>& getPhiImages();
    std::vector<PooledImage>& getPsiPooled();
    std::vector<PooledImage>& getPhiPooled();
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

    // Helper functions for pooling.
    void clearPooled();
    void poolAllImages();

    virtual ~KBMOSearch() {};

private:
    void search(bool useGpu, int aSteps, int vSteps, float minAngle,
            float maxAngle, float minVelocity, float maxVelocity, int minObservations);
    std::vector<trajRegion> resSearch(float xVel, float yVel,
            float radius, int minObservations, float minLH);
    std::vector<trajRegion> resSearchGPU(float xVel, float yVel,
            float radius, int minObservations, float minLH);
    void cpuConvolve();
    void gpuConvolve();
    void removeObjectFromImages(trajRegion& t);
    void saveImages(const std::string& path);
    void createInterleavedPsiPhi();
    void gpuSearch(int minObservations);
    void gpuSearchFilter(int minObservations); 
    void sortResults();
    std::vector<float> createCurves(trajectory t, std::vector<RawImage*> imgs);

    // Creates list of trajectories to search.
    void createSearchList(int angleSteps, int veloctiySteps, float minAngle,
                          float maxAngle, float minVelocity, float maxVelocity);

    // Helper functions for the pooled data.
    void repoolArea(trajRegion& t);
    float maxMasked(float pixel, float previousMax);
    float minMasked(float pixel, float previousMin);

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
    float sigmaGCoeff;
    float minLH;
    std::vector<float> centralMomLims;
    std::vector<float> percentiles;
    ImageStack stack;
    PointSpreadFunc psf;
    PointSpreadFunc psfSQ;
    std::vector<trajectory> searchList;
    std::vector<RawImage> psiImages;
    std::vector<RawImage> phiImages;
    std::vector<PooledImage> pooledPsi;
    std::vector<PooledImage> pooledPhi;
    std::vector<float> interleavedPsiPhi;
    std::vector<trajectory> results;

    // Variables for the timer.
    std::chrono::time_point<std::chrono::system_clock> tStart, tEnd;
    std::chrono::duration<double> tDelta;
};

} /* namespace kbmod */

#endif /* KBMODSEARCH_H_ */
