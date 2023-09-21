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
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <assert.h>
#include <float.h>
#include "common.h"
#include "ImageStack.h"
#include "PointSpreadFunc.h"

namespace search {

class KBMOSearch {
public:
    KBMOSearch(ImageStack& imstack);

    int numImages() const { return stack.imgCount(); }
    const ImageStack& getImageStack() const { return stack; }

    void setDebug(bool d);

    // The primary search functions.
    void enableGPUSigmaGFilter(std::vector<float> pyPercentiles, float pysigmag_coeff, float pymin_lh);
    void enableCorr(std::vector<float> pyBaryCorrCoeff);
    void enableGPUEncoding(int psi_num_bytes, int phi_num_bytes);

    void setStartBoundsX(int x_min, int x_max);
    void setStartBoundsY(int y_min, int y_max);

    void search(int aSteps, int vSteps, float minAngle, float maxAngle, float minVelocity, float maxVelocity,
                int min_observations);

    // Gets the vector of result trajectories.
    std::vector<trajectory> getResults(int start, int end);

    // Get the predicted (pixel) positions for a given trajectory.
    PixelPos getTrajPos(const trajectory& t, int i) const;
    std::vector<PixelPos> getMultTrajPos(trajectory& t) const;

    // Filters the results based on various parameters.
    void filterResults(int min_observations);
    void filterResultsLH(float min_lh);

    // Functions for creating science stamps for filtering, visualization, etc. User can specify
    // the radius of the stamp, whether to interpolate among pixels, whether to keep NO_DATA values
    // or replace them with zero, and what indices to use.
    // The indices to use are indicated by use_index: a vector<bool> indicating whether to use
    // each time step. An empty (size=0) vector will use all time steps.
    std::vector<RawImage> scienceStamps(const trajectory& trj, int radius, bool interpolate,
                                        bool keep_no_data, const std::vector<bool>& use_index);
    std::vector<RawImage> scienceStampsForViz(const trajectory& t, int radius);
    RawImage medianScienceStamp(const trajectory& trj, int radius, const std::vector<bool>& use_index);
    RawImage meanScienceStamp(const trajectory& trj, int radius, const std::vector<bool>& use_index);
    RawImage summedScienceStamp(const trajectory& trj, int radius, const std::vector<bool>& use_index);

    // Compute a mean or summed stamp for each trajectory on the GPU or CPU.
    // The GPU implementation is slower for small numbers of trajectories (< 500), but performs
    // relatively better as the number of trajectories increases. If filtering is applied then
    // the code will return a 1x1 image with NO_DATA to represent each filtered image.
    std::vector<RawImage> coaddedScienceStamps(std::vector<trajectory>& t_array,
                                               std::vector<std::vector<bool> >& use_index_vect,
                                               const StampParameters& params, bool use_cpu);

    // Function to do the actual stamp filtering.
    bool filterStamp(const RawImage& img, const StampParameters& params);

    // Getters for the Psi and Phi data.
    std::vector<RawImage>& getPsiImages();
    std::vector<RawImage>& getPhiImages();
    std::vector<float> psiCurves(trajectory& t);
    std::vector<float> phiCurves(trajectory& t);

    // Save internal data products to a file.
    void savePsiPhi(const std::string& path);

    // Helper functions for computing Psi and Phi.
    void preparePsiPhi();

    // Helper functions for testing.
    void setResults(const std::vector<trajectory>& new_results);

    virtual ~KBMOSearch(){};

protected:
    void saveImages(const std::string& path);
    void sortResults();
    std::vector<float> createCurves(trajectory t, const std::vector<RawImage>& imgs);

    // Fill an interleaved vector for the GPU functions.
    void fillPsiAndPhiVects(const std::vector<RawImage>& psiImgs, const std::vector<RawImage>& phiImgs,
                            std::vector<float>* psiVect, std::vector<float>* phiVect);

    // Set the parameter min/max/scale from the psi/phi/other images.
    std::vector<scaleParameters> computeImageScaling(const std::vector<RawImage>& vect,
                                                     int encoding_bytes) const;

    // Functions to create and access stamps around proposed trajectories or
    // regions. Used to visualize the results.
    // This function replaces NO_DATA with a value of 0.0.
    std::vector<RawImage> createStamps(trajectory t, int radius, const std::vector<RawImage*>& imgs,
                                       bool interpolate);

    // Creates list of trajectories to search.
    void createSearchList(int angleSteps, int veloctiySteps, float minAngle, float maxAngle,
                          float minVelocity, float maxVelocity);

    std::vector<RawImage> coaddedScienceStampsGPU(std::vector<trajectory>& t_array,
                                                  std::vector<std::vector<bool> >& use_index_vect,
                                                  const StampParameters& params);

    std::vector<RawImage> coaddedScienceStampsCPU(std::vector<trajectory>& t_array,
                                                  std::vector<std::vector<bool> >& use_index_vect,
                                                  const StampParameters& params);
    // Helper functions for timing operations of the search.
    void startTimer(const std::string& message);
    void endTimer();

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

    // Parameters for the GPU search.
    SearchParameters params;

    // Parameters to do barycentric corrections.
    bool use_corr;
    std::vector<BaryCorrection> bary_corrs;
};

} /* namespace search */

#endif /* KBMODSEARCH_H_ */
