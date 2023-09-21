/*
 * KBMOSearch.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbmod-usr
 */

#include "KBMOSearch.h"

namespace search {

#ifdef HAVE_CUDA
extern "C" void deviceSearchFilter(int num_images, int width, int height, float* psi_vect, float* phi_vect,
                                   PerImageData img_data, SearchParameters params, int num_trajectories,
                                   trajectory* trj_to_search, int num_results, trajectory* best_results);

void deviceGetCoadds(ImageStack& stack, PerImageData image_data, int num_trajectories,
                     trajectory* trajectories, StampParameters params,
                     std::vector<std::vector<bool> >& use_index_vect, float* results);
#endif

KBMOSearch::KBMOSearch(ImageStack& imstack) : stack(imstack) {
    maxResultCount = 100000;
    debugInfo = false;
    psiPhiGenerated = false;

    // Default the thresholds.
    params.min_observations = 0;
    params.min_lh = 0.0;

    // Default filtering arguments.
    params.do_sigmag_filter = false;
    params.sgl_L = 0.25;
    params.sgl_H = 0.75;
    params.sigmag_coeff = -1.0;

    // Default the encoding parameters.
    params.psi_num_bytes = -1;
    params.phi_num_bytes = -1;

    // Default pixel starting bounds.
    params.x_start_min = 0;
    params.x_start_max = stack.getWidth();
    params.y_start_min = 0;
    params.y_start_max = stack.getHeight();

    // Set default values for the barycentric correction.
    bary_corrs = std::vector<BaryCorrection>(stack.imgCount());
    params.use_corr = false;
    use_corr = false;

    params.debug = false;
}

void KBMOSearch::setDebug(bool d) {
    debugInfo = d;
    params.debug = d;
}

void KBMOSearch::enableCorr(std::vector<float> pyBaryCorrCoeff) {
    use_corr = true;
    params.use_corr = true;
    for (int i = 0; i < stack.imgCount(); i++) {
        int j = i * 6;
        bary_corrs[i].dx = pyBaryCorrCoeff[j];
        bary_corrs[i].dxdx = pyBaryCorrCoeff[j + 1];
        bary_corrs[i].dxdy = pyBaryCorrCoeff[j + 2];
        bary_corrs[i].dy = pyBaryCorrCoeff[j + 3];
        bary_corrs[i].dydx = pyBaryCorrCoeff[j + 4];
        bary_corrs[i].dydy = pyBaryCorrCoeff[j + 5];
    }
}

void KBMOSearch::enableGPUSigmaGFilter(std::vector<float> pyPercentiles, float pysigmag_coeff,
                                       float pymin_lh) {
    params.do_sigmag_filter = true;
    params.sgl_L = pyPercentiles[0];
    params.sgl_H = pyPercentiles[1];
    params.sigmag_coeff = pysigmag_coeff;
    params.min_lh = pymin_lh;
}

void KBMOSearch::enableGPUEncoding(int pypsi_num_bytes, int pyphi_num_bytes) {
    // Make sure the encoding is one of the supported options.
    // Otherwise use default float (aka no encoding).
    if (pypsi_num_bytes == 1 || pypsi_num_bytes == 2) {
        params.psi_num_bytes = pypsi_num_bytes;
    } else {
        params.psi_num_bytes = -1;
    }
    if (pyphi_num_bytes == 1 || pyphi_num_bytes == 2) {
        params.phi_num_bytes = pyphi_num_bytes;
    } else {
        params.phi_num_bytes = -1;
    }
}

void KBMOSearch::setStartBoundsX(int x_min, int x_max) {
    params.x_start_min = x_min;
    params.x_start_max = x_max;
}

void KBMOSearch::setStartBoundsY(int y_min, int y_max) {
    params.y_start_min = y_min;
    params.y_start_max = y_max;
}

void KBMOSearch::search(int aSteps, int vSteps, float minAngle, float maxAngle, float minVelocity,
                        float maxVelocity, int min_observations) {
    preparePsiPhi();
    createSearchList(aSteps, vSteps, minAngle, maxAngle, minVelocity, maxVelocity);

    startTimer("Creating psi/phi buffers");
    std::vector<float> psiVect;
    std::vector<float> phiVect;
    fillPsiAndPhiVects(psiImages, phiImages, &psiVect, &phiVect);
    endTimer();

    // Create a data stucture for the per-image data.
    PerImageData img_data;
    img_data.num_images = stack.imgCount();
    img_data.image_times = stack.getTimesDataRef();
    if (params.use_corr) img_data.bary_corrs = &bary_corrs[0];

    // Compute the encoding parameters for psi and phi if needed.
    // Vectors need to be created outside the if so they stay in scope.
    std::vector<scaleParameters> psiScaleVect;
    std::vector<scaleParameters> phiScaleVect;
    if (params.psi_num_bytes > 0) {
        psiScaleVect = computeImageScaling(psiImages, params.psi_num_bytes);
        img_data.psi_params = psiScaleVect.data();
    }
    if (params.phi_num_bytes > 0) {
        phiScaleVect = computeImageScaling(phiImages, params.phi_num_bytes);
        img_data.phi_params = phiScaleVect.data();
    }

    // Allocate a vector for the results.
    int num_search_pixels =
            ((params.x_start_max - params.x_start_min) * (params.y_start_max - params.y_start_min));
    int max_results = num_search_pixels * RESULTS_PER_PIXEL;
    if (debugInfo) {
        std::cout << "Searching X=[" << params.x_start_min << ", " << params.x_start_max << "]"
                  << " Y=[" << params.y_start_min << ", " << params.y_start_max << "]\n";
        std::cout << "Allocating space for " << max_results << " results.\n";
    }
    results = std::vector<trajectory>(max_results);
    if (debugInfo) std::cout << searchList.size() << " trajectories... \n" << std::flush;

    // Set the minimum number of observations.
    params.min_observations = min_observations;

    // Do the actual search on the GPU.
    startTimer("Searching");
#ifdef HAVE_CUDA
    deviceSearchFilter(stack.imgCount(), stack.getWidth(), stack.getHeight(), psiVect.data(), phiVect.data(),
                       img_data, params, searchList.size(), searchList.data(), max_results, results.data());
#else
    throw std::runtime_error("Non-GPU search is not implemented.");
#endif
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
        params.min_val = bnds[0];
        params.max_val = bnds[1];

        // Increase width to avoid divide by zero.
        float width = (params.max_val - params.min_val);
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
        psiImages[i].saveToFile(path + "/psi/PSI" + number + ".fits");
        phiImages[i].saveToFile(path + "/phi/PHI" + number + ".fits");
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
            searchList[a * velocitySteps + v].x_vel = cos(angles[a]) * velocities[v];
            searchList[a * velocitySteps + v].y_vel = sin(angles[a]) * velocities[v];
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

    int num_pixels = psiImgs[0].getNPixels();
    for (int i = 0; i < num_images; ++i) {
        assert(psiImgs[i].getNPixels() == num_pixels);
        assert(phiImgs[i].getNPixels() == num_pixels);
    }

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

std::vector<RawImage> KBMOSearch::scienceStamps(const trajectory& trj, int radius, bool interpolate,
                                                bool keep_no_data, const std::vector<bool>& use_index) {
    if (use_index.size() > 0 && use_index.size() != stack.imgCount()) {
        throw std::runtime_error("Wrong size use_index passed into scienceStamps()");
    }
    bool use_all_stamps = use_index.size() == 0;

    std::vector<RawImage> stamps;
    int num_times = stack.imgCount();
    for (int i = 0; i < num_times; ++i) {
        if (use_all_stamps || use_index[i]) {
            PixelPos pos = getTrajPos(trj, i);
            RawImage& img = stack.getSingleImage(i).getScience();
            stamps.push_back(img.createStamp(pos.x, pos.y, radius, interpolate, keep_no_data));
        }
    }
    return stamps;
}

// For stamps used for visualization we interpolate the pixel values, replace
// NO_DATA tages with zeros, and return all the stamps (regardless of whether
// individual timesteps have been filtered).
std::vector<RawImage> KBMOSearch::scienceStampsForViz(const trajectory& t, int radius) {
    std::vector<bool> empty_vect;
    return scienceStamps(t, radius, true /*=interpolate*/, false /*=keep_no_data*/, empty_vect);
}

// For creating coadded stamps, we do not interpolate the pixel values and keep
// NO_DATA tagged (so we can filter it out of mean/median).
RawImage KBMOSearch::medianScienceStamp(const trajectory& trj, int radius,
                                        const std::vector<bool>& use_index) {
    return createMedianImage(
            scienceStamps(trj, radius, false /*=interpolate*/, true /*=keep_no_data*/, use_index));
}

// For creating coadded stamps, we do not interpolate the pixel values and keep
// NO_DATA tagged (so we can filter it out of mean/median).
RawImage KBMOSearch::meanScienceStamp(const trajectory& trj, int radius, const std::vector<bool>& use_index) {
    return createMeanImage(
            scienceStamps(trj, radius, false /*=interpolate*/, true /*=keep_no_data*/, use_index));
}

// For creating summed stamps, we do not interpolate the pixel values and replace NO_DATA
// with zero (which is the same as filtering it out for the sum).
RawImage KBMOSearch::summedScienceStamp(const trajectory& trj, int radius,
                                        const std::vector<bool>& use_index) {
    return createSummedImage(
            scienceStamps(trj, radius, false /*=interpolate*/, false /*=keep_no_data*/, use_index));
}

bool KBMOSearch::filterStamp(const RawImage& img, const StampParameters& params) {
    // Allocate space for the coadd information and initialize to zero.
    const int stamp_width = 2 * params.radius + 1;
    const int stamp_ppi = stamp_width * stamp_width;
    const std::vector<float>& pixels = img.getPixels();

    // Filter on the peak's position.
    PixelPos pos = img.findPeak(true);
    if ((abs(pos.x - params.radius) >= params.peak_offset_x) ||
        (abs(pos.y - params.radius) >= params.peak_offset_y)) {
        return true;
    }

    // Filter on the percentage of flux in the central pixel.
    if (params.center_thresh > 0.0) {
        const std::vector<float>& pixels = img.getPixels();
        float center_val = pixels[(int)pos.y * stamp_width + (int)pos.x];
        float pixel_sum = 0.0;
        for (int p = 0; p < stamp_ppi; ++p) {
            pixel_sum += pixels[p];
        }

        if (center_val / pixel_sum < params.center_thresh) {
            return true;
        }
    }

    // Filter on the image moments.
    ImageMoments moments = img.findCentralMoments();
    if ((fabs(moments.m01) >= params.m01_limit) || (fabs(moments.m10) >= params.m10_limit) ||
        (fabs(moments.m11) >= params.m11_limit) || (moments.m02 >= params.m02_limit) ||
        (moments.m20 >= params.m20_limit)) {
        return true;
    }

    return false;
}

std::vector<RawImage> KBMOSearch::coaddedScienceStamps(std::vector<trajectory>& t_array,
                                                       std::vector<std::vector<bool> >& use_index_vect,
                                                       const StampParameters& params, bool use_gpu) {
    if (use_gpu) {
#ifdef HAVE_CUDA
        return coaddedScienceStampsGPU(t_array, use_index_vect, params);
#else
        print("WARNING: GPU is not enabled. Performing co-adds on the CPU.");
#endif
    }
    return coaddedScienceStampsCPU(t_array, use_index_vect, params);
}

std::vector<RawImage> KBMOSearch::coaddedScienceStampsCPU(std::vector<trajectory>& t_array,
                                                          std::vector<std::vector<bool> >& use_index_vect,
                                                          const StampParameters& params) {
    const int num_trajectories = t_array.size();
    std::vector<RawImage> results(num_trajectories);
    std::vector<float> empty_pixels(1, NO_DATA);

    for (int i = 0; i < num_trajectories; ++i) {
        std::vector<RawImage> stamps =
                scienceStamps(t_array[i], params.radius, false, true, use_index_vect[i]);

        RawImage coadd(1, 1);
        switch (params.stamp_type) {
            case STAMP_MEDIAN:
                coadd = createMedianImage(stamps);
                break;
            case STAMP_MEAN:
                coadd = createMeanImage(stamps);
                break;
            case STAMP_SUM:
                coadd = createSummedImage(stamps);
                break;
            default:
                throw std::runtime_error("Invalid stamp coadd type.");
        }

        // Do the filtering if needed.
        if (params.do_filtering && filterStamp(coadd, params)) {
            results[i] = RawImage(1, 1, empty_pixels);
        } else {
            results[i] = coadd;
        }
    }

    return results;
}

std::vector<RawImage> KBMOSearch::coaddedScienceStampsGPU(std::vector<trajectory>& t_array,
                                                          std::vector<std::vector<bool> >& use_index_vect,
                                                          const StampParameters& params) {
    // Right now only limited stamp sizes are allowed.
    if (2 * params.radius + 1 > MAX_STAMP_EDGE || params.radius <= 0) {
        throw std::runtime_error("Invalid Radius.");
    }

    const int num_images = stack.imgCount();
    const int width = stack.getWidth();
    const int height = stack.getHeight();

    // Create a data stucture for the per-image data.
    PerImageData img_data;
    img_data.num_images = num_images;
    img_data.image_times = stack.getTimesDataRef();

    // Allocate space for the results.
    const int num_trajectories = t_array.size();
    const int stamp_width = 2 * params.radius + 1;
    const int stamp_ppi = stamp_width * stamp_width;
    std::vector<float> stamp_data(stamp_ppi * num_trajectories);

// Do the co-adds.
#ifdef HAVE_CUDA
    deviceGetCoadds(stack, img_data, num_trajectories, t_array.data(), params, use_index_vect,
                    stamp_data.data());
#else
    throw std::runtime_error("Non-GPU co-adds is not implemented.");
#endif

    // Copy the stamps into RawImages and do the filtering.
    std::vector<RawImage> results(num_trajectories);
    std::vector<float> current_pixels(stamp_ppi, 0.0);
    std::vector<float> empty_pixels(1, NO_DATA);
    for (int t = 0; t < num_trajectories; ++t) {
        // Copy the data into a single RawImage.
        int offset = t * stamp_ppi;
        for (unsigned p = 0; p < stamp_ppi; ++p) {
            current_pixels[p] = stamp_data[offset + p];
        }
        RawImage current_image = RawImage(stamp_width, stamp_width, current_pixels);

        if (params.do_filtering && filterStamp(current_image, params)) {
            results[t] = RawImage(1, 1, empty_pixels);
        } else {
            results[t] = RawImage(stamp_width, stamp_width, current_pixels);
        }
    }
    return results;
}

std::vector<RawImage> KBMOSearch::createStamps(trajectory t, int radius, const std::vector<RawImage*>& imgs,
                                               bool interpolate) {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");
    std::vector<RawImage> stamps;
    for (int i = 0; i < imgs.size(); ++i) {
        PixelPos pos = getTrajPos(t, i);
        stamps.push_back(imgs[i]->createStamp(pos.x, pos.y, radius, interpolate, false));
    }
    return stamps;
}

PixelPos KBMOSearch::getTrajPos(const trajectory& t, int i) const {
    float time = stack.getTimes()[i];
    if (use_corr) {
        return {t.x + time * t.x_vel + bary_corrs[i].dx + t.x * bary_corrs[i].dxdx + t.y * bary_corrs[i].dxdy,
                t.y + time * t.y_vel + bary_corrs[i].dy + t.x * bary_corrs[i].dydx +
                        t.y * bary_corrs[i].dydy};
    } else {
        return {t.x + time * t.x_vel, t.y + time * t.y_vel};
    }
}

std::vector<PixelPos> KBMOSearch::getMultTrajPos(trajectory& t) const {
    std::vector<PixelPos> results;
    int num_times = stack.imgCount();
    for (int i = 0; i < num_times; ++i) {
        PixelPos pos = getTrajPos(t, i);
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
        float pix_val;
        if (use_corr) {
            PixelPos pos = getTrajPos(t, i);
            pix_val = imgs[i].getPixel(int(pos.x + 0.5), int(pos.y + 0.5));
        }
        /* Does not use getTrajPos to be backwards compatible with Hits_Rerun */
        else {
            pix_val = imgs[i].getPixel(t.x + int(times[i] * t.x_vel + 0.5),
                                       t.y + int(times[i] * t.y_vel + 0.5));
        }
        if (pix_val == NO_DATA) pix_val = 0.0;
        lightcurve.push_back(pix_val);
    }
    return lightcurve;
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

void KBMOSearch::filterResults(int min_observations) {
    results.erase(std::remove_if(results.begin(), results.end(),
                                 std::bind([](trajectory t, int cutoff) { return t.obs_count < cutoff; },
                                           std::placeholders::_1, min_observations)),
                  results.end());
}

void KBMOSearch::filterResultsLH(float min_lh) {
    results.erase(std::remove_if(results.begin(), results.end(),
                                 std::bind([](trajectory t, float cutoff) { return t.lh < cutoff; },
                                           std::placeholders::_1, min_lh)),
                  results.end());
}

std::vector<trajectory> KBMOSearch::getResults(int start, int count) {
    if (start + count >= results.size()) {
        count = results.size() - start;
    }
    if (start < 0) throw std::runtime_error("start must be 0 or greater");
    return std::vector<trajectory>(results.begin() + start, results.begin() + start + count);
}

// This function is used only for testing by injecting known result trajectories.
void KBMOSearch::setResults(const std::vector<trajectory>& new_results) { results = new_results; }

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
