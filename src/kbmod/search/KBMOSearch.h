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
#include "psf.h"

namespace search {

class KBMOSearch {
public:
    KBMOSearch(ImageStack& imstack);

    int numImages() const { return stack.imgCount(); }
    const ImageStack& getImageStack() const { return stack; }

    void setDebug(bool d);

    // The primary search functions.
    void enableGPUSigmaGFilter(std::vector<float> percentiles, float sigmag_coeff, float min_lh);
    void enableCorr(std::vector<float> bary_corr_coeff);
    void enableGPUEncoding(int psi_num_bytes, int phi_num_bytes);

    void setStartBoundsX(int x_min, int x_max);
    void setStartBoundsY(int y_min, int y_max);

    void search(int a_steps, int v_steps, float min_angle, float max_angle, float min_velocity,
                float max_velocity, int min_observations);

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
    void fillPsiAndphi_vects(const std::vector<RawImage>& psi_imgs, const std::vector<RawImage>& phi_imgs,
                            std::vector<float>* psi_vect, std::vector<float>* phi_vect);

    // Set the parameter min/max/scale from the psi/phi/other images.
    std::vector<scaleParameters> computeImageScaling(const std::vector<RawImage>& vect,
                                                     int encoding_bytes) const;

    // Functions to create and access stamps around proposed trajectories or
    // regions. Used to visualize the results.
    // This function replaces NO_DATA with a value of 0.0.
    std::vector<RawImage> create_stamps(trajectory t, int radius, const std::vector<RawImage*>& imgs,
                                       bool interpolate);

    // Creates list of trajectories to search.
    void createSearchList(int angle_steps, int velocity_steps, float min_ang, float max_ang,
                          float min_vel, float max_vel);

    std::vector<RawImage> coaddedScienceStampsGPU(std::vector<trajectory>& t_array,
                                                  std::vector<std::vector<bool> >& use_index_vect,
                                                  const StampParameters& params);

    std::vector<RawImage> coaddedScienceStampsCPU(std::vector<trajectory>& t_array,
                                                  std::vector<std::vector<bool> >& use_index_vect,
                                                  const StampParameters& params);
    // Helper functions for timing operations of the search.
    void startTimer(const std::string& message);
    void endTimer();

    unsigned max_result_count;
    bool psi_phi_generated;
    bool debug_info;
    ImageStack stack;
    std::vector<trajectory> search_list;
    std::vector<RawImage> psi_images;
    std::vector<RawImage> phi_images;
    std::vector<trajectory> results;

    // Variables for the timer.
    std::chrono::time_point<std::chrono::system_clock> t_start, t_end;
    std::chrono::duration<double> t_delta;

    // Parameters for the GPU search.
    SearchParameters params;

    // Parameters to do barycentric corrections.
    bool use_corr;
    std::vector<BaryCorrection> bary_corrs;
};

} /* namespace search */

#endif /* KBMODSEARCH_H_ */
