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
#include "RawImage.h"
#include "common.h"

namespace search {

class LayeredImage {
public:
    explicit LayeredImage(std::string path, const PointSpreadFunc& psf);
    explicit LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk, float time,
                          const PointSpreadFunc& psf);
    explicit LayeredImage(std::string name, int w, int h, float noiseStDev, float pixelVariance, double time,
                          const PointSpreadFunc& psf);
    explicit LayeredImage(std::string name, int w, int h, float noiseStDev, float pixelVariance, double time,
                          const PointSpreadFunc& psf, int seed);

    // Set an image specific point spread function.
    void setPSF(const PointSpreadFunc& psf);
    const PointSpreadFunc& getPSF() const { return psf; }
    const PointSpreadFunc& getPSFSQ() const { return psfSQ; }

    // Basic getter functions for image data.
    std::string getName() const { return fileName; }
    unsigned getWidth() const { return width; }
    unsigned getHeight() const { return height; }
    unsigned getPPI() const { return width * height; }
    double getTime() const { return captureTime; }

    // Basic setter functions.
    void setTime(double timestamp) { captureTime = timestamp; }

    // Getter functions for the data in the individual layers.
    RawImage& getScience() { return science; }
    RawImage& getMask() { return mask; }
    RawImage& getVariance() { return variance; }

    // Get pointers to the raw pixel arrays.
    float* getSDataRef() { return science.getDataRef(); }
    float* getVDataRef() { return variance.getDataRef(); }
    float* getMDataRef() { return mask.getDataRef(); }

    // Applies the mask functions to each of the science and variance layers.
    void applyMaskFlags(int flag, const std::vector<int>& exceptions);
    void applyGlobalMask(const RawImage& globalMask);
    void applyMaskThreshold(float thresh);
    void growMask(int steps);

    // Subtracts a template image from the science layer.
    void subtractTemplate(const RawImage& subTemplate);

    // Adds an (artificial) object to the image (science) data.
    void addObject(float x, float y, float flux);

    // Saves the data in each later to a file.
    void saveLayers(const std::string& path);
    void saveSci(const std::string& path);
    void saveMask(const std::string& path);
    void saveVar(const std::string& path);

    // Setter functions for the individual layers.
    void setScience(RawImage& im);
    void setMask(RawImage& im);
    void setVariance(RawImage& im);

    void convolvePSF();
    virtual ~LayeredImage(){};

    // Generate psi and phi images from the science and variance layers.
    RawImage generatePsiImage();
    RawImage generatePhiImage();

private:
    void readHeader(const std::string& filePath);
    void loadLayers(const std::string& filePath);
    void readFitsImg(const char* name, float* target);
    void checkDims(RawImage& im);

    std::string fileName;
    unsigned width;
    unsigned height;
    double captureTime;

    PointSpreadFunc psf;
    PointSpreadFunc psfSQ;
    RawImage science;
    RawImage mask;
    RawImage variance;
};

} /* namespace search */

#endif /* LAYEREDIMAGE_H_ */
