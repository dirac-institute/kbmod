/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace search {

extern "C" void deviceSearchFilter(int imageCount, int width, int height, float* psiVect, float* phiVect,
                                   perImageData img_data, searchParameters params, int trajCount,
                                   trajectory* trajectoriesToSearch, int resultsCount,
                                   trajectory* bestTrajects);

void deviceGetCoadds(ImageStack& stack, perImageData image_data, int num_trajectories,
                     trajectory *trajectories, stampParameters params,
                     std::vector<std::vector<bool> >& use_index_vect, float* results);


KBMOSearch::KBMOSearch(ImageStack& imstack) : stack(imstack) {
    maxResultCount = 100000;
    debugInfo = false;
    psiPhiGenerated = false;

    // Default the thresholds.
    params.minObservations = 0;
    params.minLH = 0.0;

    // Default filtering arguments.
    params.do_sigmag_filter = false;
    params.sGL_L = 0.25;
    params.sGL_H = 0.75;
    params.sigmaGCoeff = -1.0;

    // Default the encoding parameters.
    params.psiNumBytes = -1;
    params.phiNumBytes = -1;

    // Set default values for the barycentric correction.
    baryCorrs = std::vector<baryCorrection>(stack.imgCount());
    params.useCorr = false;
    useCorr = false;
}

void KBMOSearch::enableCorr(std::vector<float> pyBaryCorrCoeff) {
    useCorr = true;
    params.useCorr = true;
    for (int i = 0; i < stack.imgCount(); i++) {
        int j = i * 6;
        baryCorrs[i].dx = pyBaryCorrCoeff[j];
        baryCorrs[i].dxdx = pyBaryCorrCoeff[j + 1];
        baryCorrs[i].dxdy = pyBaryCorrCoeff[j + 2];
        baryCorrs[i].dy = pyBaryCorrCoeff[j + 3];
        baryCorrs[i].dydx = pyBaryCorrCoeff[j + 4];
        baryCorrs[i].dydy = pyBaryCorrCoeff[j + 5];
    }
}

void KBMOSearch::enableGPUSigmaGFilter(std::vector<float> pyPercentiles, float pySigmaGCoeff, float pyMinLH) {
    params.do_sigmag_filter = true;
    params.sGL_L = pyPercentiles[0];
    params.sGL_H = pyPercentiles[1];
    params.sigmaGCoeff = pySigmaGCoeff;
    params.minLH = pyMinLH;
}

void KBMOSearch::enableGPUEncoding(int pyPsiNumBytes, int pyPhiNumBytes) {
    // Make sure the encoding is one of the supported options.
    // Otherwise use default float (aka no encoding).
    if (pyPsiNumBytes == 1 || pyPsiNumBytes == 2) {
        params.psiNumBytes = pyPsiNumBytes;
    } else {
        params.psiNumBytes = -1;
    }
    if (pyPhiNumBytes == 1 || pyPhiNumBytes == 2) {
        params.phiNumBytes = pyPhiNumBytes;
    } else {
        params.phiNumBytes = -1;
    }
}

void KBMOSearch::search(int aSteps, int vSteps, float minAngle, float maxAngle, float minVelocity,
                        float maxVelocity, int minObservations) {
    preparePsiPhi();
    createSearchList(aSteps, vSteps, minAngle, maxAngle, minVelocity, maxVelocity);

    startTimer("Creating psi/phi buffers");
    std::vector<float> psiVect;
    std::vector<float> phiVect;
    fillPsiAndPhiVects(psiImages, phiImages, &psiVect, &phiVect);
    endTimer();

    // Create a data stucture for the per-image data.
    perImageData img_data;
    img_data.numImages = stack.imgCount();
    img_data.imageTimes = stack.getTimesDataRef();
    if (params.useCorr) img_data.baryCorrs = &baryCorrs[0];

    // Compute the encoding parameters for psi and phi if needed.
    // Vectors need to be created outside the if so they stay in scope.
    std::vector<scaleParameters> psiScaleVect;
    std::vector<scaleParameters> phiScaleVect;
    if (params.psiNumBytes > 0) {
        psiScaleVect = computeImageScaling(psiImages, params.psiNumBytes);
        img_data.psiParams = psiScaleVect.data();
    }
    if (params.phiNumBytes > 0) {
        phiScaleVect = computeImageScaling(phiImages, params.phiNumBytes);
        img_data.phiParams = phiScaleVect.data();
    }

    // Allocate a vector for the results.
    results = std::vector<trajectory>(stack.getPPI() * RESULTS_PER_PIXEL);
    if (debugInfo) std::cout << searchList.size() << " trajectories... \n" << std::flush;

    // Set the minimum number of observations.
    params.minObservations = minObservations;

    // Do the actual search on the GPU.
    startTimer("Searching");
    deviceSearchFilter(stack.imgCount(), stack.getWidth(), stack.getHeight(), psiVect.data(), phiVect.data(),
                       img_data, params, searchList.size(), searchList.data(),
                       stack.getPPI() * RESULTS_PER_PIXEL, results.data());
    endTimer();

    startTimer("Sorting results");
    sortResults();
    endTimer();
}

void KBMOSearch::savePsiPhi(const std::string& path) {
    preparePsiPhi();
    saveImages(path);
}

void KBMOSearch::preparePsiPhi() {
    if (!psiPhiGenerated) {
        psiImages.clear();
        phiImages.clear();

        // Compute Phi and Psi from convolved images
        // while leaving masked pixels alone
        // Reinsert 0s for NO_DATA?
        const int num_images = stack.imgCount();
        for (int i = 0; i < num_images; ++i) {
            LayeredImage& img = stack.getSingleImage(i);
            psiImages.push_back(img.generatePsiImage());
            phiImages.push_back(img.generatePhiImage());
        }

        psiPhiGenerated = true;
    }
}

std::vector<scaleParameters> KBMOSearch::computeImageScaling(const std::vector<RawImage>& vect,
                                                             int encoding_bytes) const {
    std::vector<scaleParameters> result;

    const int num_images = vect.size();
    for (int i = 0; i < num_images; ++i) {
        scaleParameters params;
        params.scale = 1.0;

        std::array<float, 2> bnds = vect[i].computeBounds();
        params.minVal = bnds[0];
        params.maxVal = bnds[1];

        // Increase width to avoid divide by zero.
        float width = (params.maxVal - params.minVal);
        if (width < 1e-6) width = 1e-6;

        // Set the scale if we are encoding the values.
        if (encoding_bytes == 1 || encoding_bytes == 2) {
            long int num_values = (1 << (8 * encoding_bytes)) - 1;
            params.scale = width / (double)num_values;
        }

        result.push_back(params);
    }

    return result;
}

void KBMOSearch::saveImages(const std::string& path) {
    for (int i = 0; i < stack.imgCount(); ++i) {
        std::string number = std::to_string(i);
        // Add leading zeros
        number = std::string(4 - number.length(), '0') + number;
        psiImages[i].saveToFile(path + "/psi/PSI" + number + ".fits", false);
        phiImages[i].saveToFile(path + "/phi/PHI" + number + ".fits", false);
    }
}

void KBMOSearch::createSearchList(int angleSteps, int velocitySteps, float minAngle, float maxAngle,
                                  float minVelocity, float maxVelocity) {
    std::vector<float> angles(angleSteps);
    float aStepSize = (maxAngle - minAngle) / float(angleSteps);
    for (int i = 0; i < angleSteps; ++i) {
        angles[i] = minAngle + float(i) * aStepSize;
    }

    std::vector<float> velocities(velocitySteps);
    float vStepSize = (maxVelocity - minVelocity) / float(velocitySteps);
    for (int i = 0; i < velocitySteps; ++i) {
        velocities[i] = minVelocity + float(i) * vStepSize;
    }

    int trajCount = angleSteps * velocitySteps;
    searchList = std::vector<trajectory>(trajCount);
    for (int a = 0; a < angleSteps; ++a) {
        for (int v = 0; v < velocitySteps; ++v) {
            searchList[a * velocitySteps + v].xVel = cos(angles[a]) * velocities[v];
            searchList[a * velocitySteps + v].yVel = sin(angles[a]) * velocities[v];
        }
    }
}

void KBMOSearch::fillPsiAndPhiVects(const std::vector<RawImage>& psiImgs,
                                    const std::vector<RawImage>& phiImgs, std::vector<float>* psiVect,
                                    std::vector<float>* phiVect) {
    assert(psiVect != NULL);
    assert(phiVect != NULL);

    int num_images = psiImgs.size();
    assert(num_images > 0);
    assert(phiImgs.size() == num_images);

    int num_pixels = psiImgs[0].getPPI();
    assert(phiImgs[0].getPPI() == num_pixels);

    psiVect->clear();
    psiVect->reserve(num_images * num_pixels);
    phiVect->clear();
    phiVect->reserve(num_images * num_pixels);

    for (int i = 0; i < num_images; ++i) {
        const std::vector<float>& psiRef = psiImgs[i].getPixels();
        const std::vector<float>& phiRef = phiImgs[i].getPixels();
        for (unsigned p = 0; p < num_pixels; ++p) {
            psiVect->push_back(psiRef[p]);
            phiVect->push_back(phiRef[p]);
        }
    }
}

std::vector<RawImage> KBMOSearch::scienceStamps(const TrajectoryResult& trj, int radius, bool interpolate,
                                                bool keep_no_data, bool all_stamps) {
    std::vector<RawImage> stamps;
    int num_times = stack.imgCount();
    const trajectory& t = trj.get_const_trajectory();
    for (int i = 0; i < num_times; ++i) {
        if (all_stamps || trj.check_index_valid(i)) {
            pixelPos pos = getTrajPos(t, i);
            RawImage& img = stack.getSingleImage(i).getScience();
            stamps.push_back(img.createStamp(pos.x, pos.y, radius, interpolate, keep_no_data));
        }
    }
    return stamps;
}

inline std::vector<RawImage> KBMOSearch::scienceStampsForFilter(const TrajectoryResult& trj, int radius) {
    return scienceStamps(trj, radius, false, true, false);
}

inline std::vector<RawImage> KBMOSearch::scienceStampsForViz(const TrajectoryResult& trj, int radius) {
    return scienceStamps(trj, radius, true, false, true);
}

std::vector<RawImage> KBMOSearch::scienceStamps(trajectory& t, int radius) {
    TrajectoryResult trj(t, stack.imgCount());
    return scienceStampsForViz(trj, radius);
}

RawImage KBMOSearch::medianScienceStamp(const TrajectoryResult& trj, int radius, bool use_all) {
    return createMedianImage(scienceStamps(trj, radius, false, true, use_all));
}

std::vector<RawImage> KBMOSearch::medianScienceStamps(const std::vector<TrajectoryResult>& t_array,
                                                      int radius) {
    const int num_results = t_array.size();
    std::vector<RawImage> results(num_results);
    omp_set_num_threads(16);

#pragma omp parallel for
    for (int s = 0; s < num_results; ++s) {
        results[s] = medianScienceStamp(t_array[s], radius, true);
    }
    omp_set_num_threads(1);

    return (results);
}

// To be deprecated in later PR.
std::vector<RawImage> KBMOSearch::medianStamps(const std::vector<trajectory>& t_array,
                                               const std::vector<std::vector<int>>& goodIdx, int radius) {
    const int num_results = t_array.size();
    std::vector<TrajectoryResult> arr;
    for (int s = 0; s < num_results; ++s) {
        TrajectoryResult trj(t_array[s], goodIdx[s]);
        arr.push_back(trj);
    }

    return medianScienceStamps(arr, radius);
}

RawImage KBMOSearch::meanScienceStamp(const TrajectoryResult& trj, int radius, bool use_all) {
    return createMeanImage(scienceStamps(trj, radius, false, true, use_all));
}

std::vector<RawImage> KBMOSearch::meanScienceStamps(const std::vector<TrajectoryResult>& t_array,
                                                    int radius) {
    const int num_results = t_array.size();
    std::vector<RawImage> results(num_results);
    omp_set_num_threads(16);

#pragma omp parallel for
    for (int s = 0; s < num_results; ++s) {
        results[s] = meanScienceStamp(t_array[s], radius, true);
    }
    omp_set_num_threads(1);

    return (results);
}

std::vector<RawImage> KBMOSearch::coaddedScienceStampsGPU(std::vector<trajectory>& t_array,
                                                          std::vector<std::vector<bool> >& use_index_vect,
                                                          const stampParameters& params) {
    // Right now only limited stamp sizes are allowed.
    if (2 * params.radius + 1 > MAX_STAMP_EDGE || params.radius <= 0) {
        throw std::runtime_error("Invalid Radius.");
    }

    const int num_images = stack.imgCount();
    const int width = stack.getWidth();
    const int height = stack.getHeight();

    // Create a data stucture for the per-image data.
    perImageData img_data;
    img_data.numImages = num_images;
    img_data.imageTimes = stack.getTimesDataRef();

    // Allocate space for the results.
    const int num_trajectories = t_array.size();
    const int stamp_width = 2 * params.radius + 1;
    const int stamp_ppi = stamp_width * stamp_width;
    std::vector<float> stamp_data(stamp_ppi * num_trajectories);

    // Do the co-adds.
    deviceGetCoadds(stack, img_data, num_trajectories, t_array.data(), params,
                    use_index_vect, stamp_data.data()); 

    // Copy the stamps into RawImages
    std::vector<RawImage> results(num_trajectories);
    std::vector<float> current_pixels(stamp_ppi, 0.0);
    std::vector<float> empty_pixels(1, NO_DATA);
    for (int t = 0; t < num_trajectories; ++t) {
        bool all_no_data = true;
        int offset = t * stamp_ppi;
        for (unsigned p = 0; p < stamp_ppi; ++p) {
            current_pixels[p] = stamp_data[offset + p];
            all_no_data = all_no_data && (stamp_data[offset + p] == NO_DATA);
        }

        if (all_no_data && params.do_filtering) {
            results[t] = RawImage(1, 1, empty_pixels);
        } else {
            results[t] = RawImage(stamp_width, stamp_width, current_pixels);
        }
    }
    return results;
}

std::vector<RawImage> KBMOSearch::coaddedScienceStampsGPU(std::vector<trajectory>& t_array,
                                                          const stampParameters& params) {
    // Use an empty vector to indicate no filtering.
    std::vector<std::vector<bool> > use_index_vect;
    return coaddedScienceStampsGPU(t_array, use_index_vect, params);
}

std::vector<RawImage> KBMOSearch::coaddedScienceStampsGPU(std::vector<TrajectoryResult>& t_array,
                                                          const stampParameters& params) {
    const int num_traj = t_array.size();
    const int num_times = stack.imgCount();
    std::vector<std::vector<bool> > use_index_vect;
    std::vector<trajectory> trjs;

    // Copy the TrajectoryResult data into a trajectory array and an integer array
    // indicating the validity of each index.
    use_index_vect.reserve(num_traj);
    trjs.reserve(num_traj);
    for (int i = 0; i < num_traj; i++) {
        trjs.push_back(t_array[i].get_trajectory());
        use_index_vect.push_back(t_array[i].get_bool_valid_array());
    }

    return coaddedScienceStampsGPU(trjs, use_index_vect, params);
}

// To be deprecated in later PR.
std::vector<RawImage> KBMOSearch::meanStamps(const std::vector<trajectory>& t_array,
                                             const std::vector<std::vector<int>>& goodIdx, int radius) {
    const int num_results = t_array.size();
    std::vector<TrajectoryResult> arr;
    for (int s = 0; s < num_results; ++s) {
        TrajectoryResult trj(t_array[s], goodIdx[s]);
        arr.push_back(trj);
    }

    return meanScienceStamps(arr, radius);
}

RawImage KBMOSearch::summedScienceStamp(const TrajectoryResult& trj, int radius, bool use_all) {
    return createSummedImage(scienceStamps(trj, radius, false, true, use_all));
}

// To be deprecated in later PR.
RawImage KBMOSearch::stackedScience(trajectory& t, int radius) {
    TrajectoryResult trj(t, stack.imgCount());
    return createSummedImage(scienceStamps(trj, radius, false, false, true));
}

std::vector<RawImage> KBMOSearch::summedScienceStamps(const std::vector<TrajectoryResult>& t_array,
                                                      int radius) {
    const int num_results = t_array.size();
    std::vector<RawImage> results(num_results);
    omp_set_num_threads(16);

#pragma omp parallel for
    for (int s = 0; s < num_results; ++s) {
        results[s] = summedScienceStamp(t_array[s], radius, true);
    }
    omp_set_num_threads(1);

    return (results);
}

// To be deprecated in later PR.
std::vector<RawImage> KBMOSearch::summedScience(const std::vector<trajectory>& t_array, int radius) {
    int numResults = t_array.size();
    std::vector<RawImage> results(numResults);

    // Build the result for each trajectory.
    omp_set_num_threads(30);
#pragma omp parallel for
    for (int s = 0; s < numResults; ++s) {
        TrajectoryResult trj(t_array[s], stack.imgCount());
        results[s] = createSummedImage(scienceStamps(trj, radius, false, false, true));
    }
    omp_set_num_threads(1);

    return (results);
}

std::vector<RawImage> KBMOSearch::createStamps(trajectory t, int radius, const std::vector<RawImage*>& imgs,
                                               bool interpolate) {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");
    std::vector<RawImage> stamps;
    for (int i = 0; i < imgs.size(); ++i) {
        pixelPos pos = getTrajPos(t, i);
        stamps.push_back(imgs[i]->createStamp(pos.x, pos.y, radius, interpolate, false));
    }
    return stamps;
}

pixelPos KBMOSearch::getTrajPos(const trajectory& t, int i) const {
    float time = stack.getTimes()[i];
    if (useCorr) {
        return computeTrajPosBC(t, time, baryCorrs[i]);
    } else {
        return computeTrajPos(t, time);
    }
}

std::vector<pixelPos> KBMOSearch::getMultTrajPos(trajectory& t) const {
    std::vector<pixelPos> results;
    int num_times = stack.imgCount();
    for (int i = 0; i < num_times; ++i) {
        pixelPos pos = getTrajPos(t, i);
        results.push_back(pos);
    }
    return results;
}

std::vector<float> KBMOSearch::createCurves(trajectory t, const std::vector<RawImage>& imgs) {
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
    for (int i = 0; i < imgSize; ++i) {
        /* Do not use getPixelInterp(), because results from createCurves must
         * be able to recover the same likelihoods as the ones reported by the
         * gpu search.*/
        float pixVal;
        if (useCorr) {
            pixelPos pos = getTrajPos(t, i);
            pixVal = imgs[i].getPixel(int(pos.x + 0.5), int(pos.y + 0.5));
        }
        /* Does not use getTrajPos to be backwards compatible with Hits_Rerun */
        else {
            pixVal = imgs[i].getPixel(t.x + int(times[i] * t.xVel + 0.5), t.y + int(times[i] * t.yVel + 0.5));
        }
        if (pixVal == NO_DATA) pixVal = 0.0;
        lightcurve.push_back(pixVal);
    }
    return lightcurve;
}

std::vector<RawImage> KBMOSearch::psiStamps(trajectory& t, int radius) {
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : psiImages) imgs.push_back(&im);
    return createStamps(t, radius, imgs, true);
}

std::vector<RawImage> KBMOSearch::phiStamps(trajectory& t, int radius) {
    preparePsiPhi();
    std::vector<RawImage*> imgs;
    for (auto& im : phiImages) imgs.push_back(&im);
    return createStamps(t, radius, imgs, true);
}

std::vector<float> KBMOSearch::psiCurves(trajectory& t) {
    /*Generate a psi lightcurve for further analysis
     *  INPUT-
     *    trajectory& t - The trajectory along which to find the lightcurve
     *  OUTPUT-
     *    std::vector<float> - A vector of the lightcurve values
     */
    preparePsiPhi();
    return createCurves(t, psiImages);
}

std::vector<float> KBMOSearch::phiCurves(trajectory& t) {
    /*Generate a phi lightcurve for further analysis
     *  INPUT-
     *    trajectory& t - The trajectory along which to find the lightcurve
     *  OUTPUT-
     *    std::vector<float> - A vector of the lightcurve values
     */
    preparePsiPhi();
    return createCurves(t, phiImages);
}

std::vector<RawImage>& KBMOSearch::getPsiImages() { return psiImages; }

std::vector<RawImage>& KBMOSearch::getPhiImages() { return phiImages; }

void KBMOSearch::sortResults() {
    __gnu_parallel::sort(results.begin(), results.end(),
                         [](trajectory a, trajectory b) { return b.lh < a.lh; });
}

void KBMOSearch::filterResults(int minObservations) {
    results.erase(std::remove_if(results.begin(), results.end(),
                                 std::bind([](trajectory t, int cutoff) { return t.obsCount < cutoff; },
                                           std::placeholders::_1, minObservations)),
                  results.end());
}

void KBMOSearch::filterResultsLH(float minLH) {
    results.erase(std::remove_if(results.begin(), results.end(),
                                 std::bind([](trajectory t, float cutoff) { return t.lh < cutoff; },
                                           std::placeholders::_1, minLH)),
                  results.end());
}

std::vector<trajectory> KBMOSearch::getResults(int start, int count) {
    if (start < 0) throw std::runtime_error("start must be 0 or greater");
    return std::vector<trajectory>(results.begin() + start, results.begin() + start + count);
}

void KBMOSearch::saveResults(const std::string& path, float portion) {
    std::ofstream file(path.c_str());
    if (file.is_open()) {
        file << "# x y xv yv likelihood flux obs_count\n";
        int writeCount = int(portion * float(results.size()));
        for (int i = 0; i < writeCount; ++i) {
            trajectory r = results[i];
            file << r.x << " " << r.y << " " << r.xVel << " " << r.yVel << " " << r.lh << " " << r.flux << " "
                 << r.obsCount << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open results file";
    }
}

void KBMOSearch::startTimer(const std::string& message) {
    if (debugInfo) {
        std::cout << message << "... " << std::flush;
        tStart = std::chrono::system_clock::now();
    }
}

void KBMOSearch::endTimer() {
    if (debugInfo) {
        tEnd = std::chrono::system_clock::now();
        tDelta = tEnd - tStart;
        std::cout << " Took " << tDelta.count() << " seconds.\n" << std::flush;
    }
}

} /* namespace search */
