/*
 * LayeredImage.h
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 *
 *  LayeredImage stores an image from a single time with different layers of
 *  data, such as science pixels, variance pixels, and mask pixels.
 */

#ifndef LAYEREDIMAGE_H_
#define LAYEREDIMAGE_H_

#include <vector>
#include <fitsio.h>
#include <iostream>
#include <string>
#include <random>
#include <assert.h>
#include <stdexcept>
#include "raw_image.h"
#include "common.h"

namespace search {

class LayeredImage {
public:
    explicit LayeredImage(std::string path, const PSF& psf);
    explicit LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk,
                          const PSF& psf);
    explicit LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                          const PSF& psf);
    explicit LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                          const PSF& psf, int seed);

    // Set an image specific point spread function.
    void setPSF(const PSF& psf);
    const PSF& getPSF() const { return psf; }

    // Basic getter functions for image data.
    std::string getName() const { return filename; }
    unsigned getWidth() const { return width; }
    unsigned getHeight() const { return height; }
    unsigned getNPixels() const { return width * height; }
    double getObstime() const { return science.get_obstime(); }
    void setObstime(double obstime) { science.set_obstime(obstime); }

    // Getter functions for the data in the individual layers.
    RawImage& getScience() { return science; }
    RawImage& getMask() { return mask; }
    RawImage& getVariance() { return variance; }

    // Get pointers to the raw pixel arrays.
    float* getSDataRef() { return science.getDataRef(); }
    float* getVDataRef() { return variance.getDataRef(); }
    float* getMDataRef() { return mask.getDataRef(); }

    // Applies the mask functions to each of the science and variance layers.
    void apply_maskFlags(int flag, const std::vector<int>& exceptions);
    void applyGlobalMask(const RawImage& global_mask);
    void apply_maskThreshold(float thresh);
    void growMask(int steps);

    // Subtracts a template image from the science layer.
    void subtractTemplate(const RawImage& sub_template);

    // Saves the data in each later to a file.
    void saveLayers(const std::string& path);

    // Setter functions for the individual layers.
    void setScience(RawImage& im);
    void setMask(RawImage& im);
    void setVariance(RawImage& im);

    // Convolve with a given PSF or the default one.
    void convolvePSF();
    void convolveGivenPSF(const PointSpreadFunc& psf);

    virtual ~LayeredImage(){};

    // Generate psi and phi images from the science and variance layers.
    RawImage generatePsiImage();
    RawImage generatePhiImage();

private:
    void checkDims(RawImage& im);

    std::string filename;
    unsigned width;
    unsigned height;

    PSF psf;
    RawImage science;
    RawImage mask;
    RawImage variance;
};

} /* namespace search */

#endif /* LAYEREDIMAGE_H_ */
