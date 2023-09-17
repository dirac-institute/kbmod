/*
 * LayeredImage.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 */

#include "LayeredImage.h"

namespace search {

LayeredImage::LayeredImage(std::string path, const PointSpreadFunc& psf) : psf(psf), psfSQ(psf) {
    psfSQ.squarePSF();

    int fBegin = path.find_last_of("/");
    int fEnd = path.find_last_of(".fits") - 4;
    fileName = path.substr(fBegin, fEnd - fBegin);

    science = RawImage(path, 1);
    mask = RawImage(path, 2);
    variance = RawImage(path, 3);

    width = science.getWidth();
    height = science.getHeight();
    if (width != variance.getWidth() or height != variance.getHeight())
        throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != mask.getWidth() or height != mask.getHeight())
        throw std::runtime_error("Science and Mask layers are not the same size.");
}

LayeredImage::LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk,
                           const PointSpreadFunc& psf)
        : psf(psf), psfSQ(psf) {
    // Get the dimensions of the science layer and check for consistency with
    // the other two layers.
    width = sci.getWidth();
    height = sci.getHeight();
    if (width != var.getWidth() or height != var.getHeight())
        throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != msk.getWidth() or height != msk.getHeight())
        throw std::runtime_error("Science and Mask layers are not the same size.");

    // Set the remaining variables.
    psfSQ.squarePSF();

    // Copy the image layers.
    science = sci;
    mask = msk;
    variance = var;
}

LayeredImage::LayeredImage(std::string name, int w, int h, float noiseStDev, float pixelVariance, double time,
                           const PointSpreadFunc& psf)
        : LayeredImage(name, w, h, noiseStDev, pixelVariance, time, psf, -1) {}

LayeredImage::LayeredImage(std::string name, int w, int h, float noiseStDev, float pixelVariance, double time,
                           const PointSpreadFunc& psf, int seed)
        : psf(psf), psfSQ(psf) {
    fileName = name;
    width = w;
    height = h;
    psfSQ.squarePSF();

    std::vector<float> rawSci(width * height);
    std::random_device r;
    std::default_random_engine generator(r());
    if (seed >= 0) {
        generator.seed(seed);
    }
    std::normal_distribution<float> distrib(0.0, noiseStDev);
    for (float& p : rawSci) p = distrib(generator);

    science = RawImage(w, h, rawSci);
    science.setObstime(time);

    mask = RawImage(w, h, std::vector<float>(w * h, 0.0));
    variance = RawImage(w, h, std::vector<float>(w * h, pixelVariance));
}

void LayeredImage::setPSF(const PointSpreadFunc& new_psf) {
    psf = new_psf;
    psfSQ = new_psf;
    psfSQ.squarePSF();
}

void LayeredImage::addObject(float x, float y, float flux) {
    const std::vector<float>& k = psf.getKernel();
    int dim = psf.getDim();
    float initialX = x - static_cast<float>(psf.getRadius());
    float initialY = y - static_cast<float>(psf.getRadius());

    int count = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            science.addPixelInterp(initialX + static_cast<float>(i), initialY + static_cast<float>(j),
                                   flux * k[count]);
            count++;
        }
    }
}

void LayeredImage::growMask(int steps) {
    science.growMask(steps);
    variance.growMask(steps);
}

void LayeredImage::convolvePSF() {
    science.convolve(psf);
    variance.convolve(psfSQ);
}

void LayeredImage::applyMaskFlags(int flags, const std::vector<int>& exceptions) {
    science.applyMask(flags, exceptions, mask);
    variance.applyMask(flags, exceptions, mask);
}

/* Mask all pixels that are not 0 in global mask */
void LayeredImage::applyGlobalMask(const RawImage& globalM) {
    science.applyMask(0xFFFFFF, {}, globalM);
    variance.applyMask(0xFFFFFF, {}, globalM);
}

void LayeredImage::applyMaskThreshold(float thresh) {
    const int numPixels = getNPixels();
    float* sciPix = science.getDataRef();
    float* varPix = variance.getDataRef();
    for (int i = 0; i < numPixels; ++i) {
        if (sciPix[i] > thresh) {
            sciPix[i] = NO_DATA;
            varPix[i] = NO_DATA;
        }
    }
}

void LayeredImage::subtractTemplate(const RawImage& subTemplate) {
    assert(getHeight() == subTemplate.getHeight() && getWidth() == subTemplate.getWidth());
    const int numPixels = getNPixels();

    float* sciPix = science.getDataRef();
    const std::vector<float>& temNPixelsx = subTemplate.getPixels();
    for (unsigned i = 0; i < numPixels; ++i) {
        if ((sciPix[i] != NO_DATA) && (temNPixelsx[i] != NO_DATA)) {
            sciPix[i] -= temNPixelsx[i];
        }
    }
}

void LayeredImage::saveLayers(const std::string& path) {
    fitsfile* fptr;
    int status = 0;
    long naxes[2] = {0, 0};
    float obstime = science.getObstime();

    fits_create_file(&fptr, (path + fileName + ".fits").c_str(), &status);

    // If we are unable to create the file, check if it already exists
    // and, if so, delete it and retry the create.
    if (status == 105) {
        status = 0;
        fits_open_file(&fptr, (path + fileName + ".fits").c_str(), READWRITE, &status);
        if (status == 0) {
            fits_delete_file(fptr, &status);
            fits_create_file(&fptr, (path + fileName + ".fits").c_str(), &status);
        }
    }

    fits_create_img(fptr, SHORT_IMG, 0, naxes, &status);
    fits_update_key(fptr, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);

    science.appendLayerToFile(path + fileName + ".fits");
    mask.appendLayerToFile(path + fileName + ".fits");
    variance.appendLayerToFile(path + fileName + ".fits");
}

void LayeredImage::setScience(RawImage& im) {
    checkDims(im);
    science = im;
}

void LayeredImage::setMask(RawImage& im) {
    checkDims(im);
    mask = im;
}

void LayeredImage::setVariance(RawImage& im) {
    checkDims(im);
    variance = im;
}

void LayeredImage::checkDims(RawImage& im) {
    if (im.getWidth() != getWidth()) throw std::runtime_error("Image width does not match");
    if (im.getHeight() != getHeight()) throw std::runtime_error("Image height does not match");
}

RawImage LayeredImage::generatePsiImage() {
    RawImage result(width, height);
    float* result_arr = result.getDataRef();
    float* sciArray = getSDataRef();
    float* varArray = getVDataRef();

    // Set each of the result pixels.
    const int num_pixels = getNPixels();
    for (int p = 0; p < num_pixels; ++p) {
        float varPix = varArray[p];
        if (varPix != NO_DATA) {
            result_arr[p] = sciArray[p] / varPix;
        } else {
            result_arr[p] = NO_DATA;
        }
    }

    // Convolve with the PSF.
    result.convolve(getPSF());

    return result;
}

RawImage LayeredImage::generatePhiImage() {
    RawImage result(width, height);
    float* result_arr = result.getDataRef();
    float* varArray = getVDataRef();

    // Set each of the result pixels.
    const int num_pixels = getNPixels();
    for (int p = 0; p < num_pixels; ++p) {
        float varPix = varArray[p];
        if (varPix != NO_DATA) {
            result_arr[p] = 1.0 / varPix;
        } else {
            result_arr[p] = NO_DATA;
        }
    }

    // Convolve with the PSF squared.
    result.convolve(getPSFSQ());

    return result;
}

} /* namespace search */
