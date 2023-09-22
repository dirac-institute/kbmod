/*
 * LayeredImage.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 */

#include "LayeredImage.h"

namespace search {

LayeredImage::LayeredImage(std::string path, const PointSpreadFunc& psf) : psf(psf) {
    int f_begin = path.find_last_of("/");
    int f_end = path.find_last_of(".fits") - 4;
    filename = path.substr(f_begin, f_end - f_begin);

    science = RawImage();
    science.loadFromFile(path, 1);
    width = science.getWidth();
    height = science.getHeight();

    mask = RawImage();
    mask.loadFromFile(path, 2);

    variance = RawImage();
    variance.loadFromFile(path, 3);

    if (width != variance.getWidth() or height != variance.getHeight())
        throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != mask.getWidth() or height != mask.getHeight())
        throw std::runtime_error("Science and Mask layers are not the same size.");
}

LayeredImage::LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk,
                           const PointSpreadFunc& psf)
        : psf(psf) {
    // Get the dimensions of the science layer and check for consistency with
    // the other two layers.
    width = sci.getWidth();
    height = sci.getHeight();
    if (width != var.getWidth() or height != var.getHeight())
        throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != msk.getWidth() or height != msk.getHeight())
        throw std::runtime_error("Science and Mask layers are not the same size.");

    // Copy the image layers.
    science = sci;
    mask = msk;
    variance = var;
}

LayeredImage::LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                           const PointSpreadFunc& psf)
        : LayeredImage(name, w, h, noise_stdev, pixel_variance, time, psf, -1) {}

LayeredImage::LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                           const PointSpreadFunc& psf, int seed)
        : psf(psf) {
    filename = name;
    width = w;
    height = h;

    std::vector<float> raw_sci(width * height);
    std::random_device r;
    std::default_random_engine generator(r());
    if (seed >= 0) {
        generator.seed(seed);
    }
    std::normal_distribution<float> distrib(0.0, noise_stdev);
    for (float& p : raw_sci) p = distrib(generator);

    science = RawImage(w, h, raw_sci);
    science.setObstime(time);

    mask = RawImage(w, h, std::vector<float>(w * h, 0.0));
    variance = RawImage(w, h, std::vector<float>(w * h, pixel_variance));
}

void LayeredImage::setPSF(const PointSpreadFunc& new_psf) {
    psf = new_psf;
}

void LayeredImage::growMask(int steps) {
    science.growMask(steps);
    variance.growMask(steps);
}

void LayeredImage::convolvePSF() {
    science.convolve(psf);

    // Square the PSF use that on the variance image.
    psf_sq = new_psf;
    psf_sq.squarePSF();
    variance.convolve(psf_sq);
}

void LayeredImage::applyMaskFlags(int flags, const std::vector<int>& exceptions) {
    science.applyMask(flags, exceptions, mask);
    variance.applyMask(flags, exceptions, mask);
}

/* Mask all pixels that are not 0 in global mask */
void LayeredImage::applyGlobalMask(const RawImage& global_mask) {
    science.applyMask(0xFFFFFF, {}, global_mask);
    variance.applyMask(0xFFFFFF, {}, global_mask);
}

void LayeredImage::applyMaskThreshold(float thresh) {
    const int num_pixels = getNPixels();
    float* sci_pixels = science.getDataRef();
    float* var_pix = variance.getDataRef();
    for (int i = 0; i < num_pixels; ++i) {
        if (sci_pixels[i] > thresh) {
            sci_pixels[i] = NO_DATA;
            var_pix[i] = NO_DATA;
        }
    }
}

void LayeredImage::subtractTemplate(const RawImage& sub_template) {
    assert(getHeight() == sub_template.getHeight() && getWidth() == sub_template.getWidth());
    const int num_pixels = getNPixels();

    float* sci_pixels = science.getDataRef();
    const std::vector<float>& tem_pixels = sub_template.getPixels();
    for (unsigned i = 0; i < num_pixels; ++i) {
        if ((sci_pixels[i] != NO_DATA) && (tem_pixels[i] != NO_DATA)) {
            sci_pixels[i] -= tem_pixels[i];
        }
    }
}

void LayeredImage::saveLayers(const std::string& path) {
    fitsfile* fptr;
    int status = 0;
    long naxes[2] = {0, 0};
    double obstime = science.getObstime();

    fits_create_file(&fptr, (path + filename + ".fits").c_str(), &status);

    // If we are unable to create the file, check if it already exists
    // and, if so, delete it and retry the create.
    if (status == 105) {
        status = 0;
        fits_open_file(&fptr, (path + filename + ".fits").c_str(), READWRITE, &status);
        if (status == 0) {
            fits_delete_file(fptr, &status);
            fits_create_file(&fptr, (path + filename + ".fits").c_str(), &status);
        }
    }

    fits_create_img(fptr, SHORT_IMG, 0, naxes, &status);
    fits_update_key(fptr, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);

    science.appendLayerToFile(path + filename + ".fits");
    mask.appendLayerToFile(path + filename + ".fits");
    variance.appendLayerToFile(path + filename + ".fits");
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
    float* sci_array = getSDataRef();
    float* var_array = getVDataRef();

    // Set each of the result pixels.
    const int num_pixels = getNPixels();
    for (int p = 0; p < num_pixels; ++p) {
        float var_pix = var_array[p];
        if (var_pix != NO_DATA) {
            result_arr[p] = sci_array[p] / var_pix;
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
    float* var_array = getVDataRef();

    // Set each of the result pixels.
    const int num_pixels = getNPixels();
    for (int p = 0; p < num_pixels; ++p) {
        float var_pix = var_array[p];
        if (var_pix != NO_DATA) {
            result_arr[p] = 1.0 / var_pix;
        } else {
            result_arr[p] = NO_DATA;
        }
    }

    // Convolve with the PSF squared.
    result.convolve(getPSFSQ());

    return result;
}

} /* namespace search */
