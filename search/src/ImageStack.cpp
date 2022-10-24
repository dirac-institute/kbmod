/*
 * ImageStack.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "ImageStack.h"

namespace kbmod {

ImageStack::ImageStack(const std::vector<std::string>& filenames, const std::vector<PointSpreadFunc>& psfs) {
    verbose = true;
    resetImages();
    loadImages(filenames, psfs);
    extractImageTimes();
    setTimeOrigin();
    globalMask = RawImage(getWidth(), getHeight());
    globalMask.setAllPix(0.0);
}

ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) {
    verbose = true;
    images = imgs;
    extractImageTimes();
    setTimeOrigin();
    globalMask = RawImage(getWidth(), getHeight());
    globalMask.setAllPix(0.0);
}

void ImageStack::loadImages(const std::vector<std::string>& fileNames,
                            const std::vector<PointSpreadFunc>& psfs) {
    const int num_files = fileNames.size();
    if (num_files == 0) {
        std::cout << "No files provided"
                  << "\n";
    }

    if (psfs.size() != num_files) throw std::runtime_error("Mismatched PSF array in ImageStack creation.");

    // Load images from file
    for (int i = 0; i < num_files; ++i) {
        images.push_back(LayeredImage(fileNames[i], psfs[i]));
        if (verbose) std::cout << "." << std::flush;
    }
    if (verbose) std::cout << "\n";
}

void ImageStack::extractImageTimes() {
    // Load image times
    imageTimes = std::vector<float>();
    for (auto& i : images) {
        imageTimes.push_back(float(i.getTime()));
    }
}

void ImageStack::setTimeOrigin() {
    // Set beginning time to 0.0
    double initialTime = imageTimes[0];
    for (auto& t : imageTimes) t = t - initialTime;
}

std::vector<LayeredImage>& ImageStack::getImages() { return images; }

unsigned ImageStack::imgCount() const { return images.size(); }

const std::vector<float>& ImageStack::getTimes() const { return imageTimes; }

float* ImageStack::getTimesDataRef() { return imageTimes.data(); }

LayeredImage& ImageStack::getSingleImage(int index) {
    if (index < 0 || index > images.size()) throw std::runtime_error("ImageStack index out of bounds.");
    return images[index];
}
    
void ImageStack::setSingleImage(int index, LayeredImage& img) {
    if (index < 0 || index > images.size()) throw std::runtime_error("ImageStack index out of bounds.");
    images[index] = img;
}

void ImageStack::setTimes(const std::vector<float>& times) {
    if (times.size() != imgCount())
        throw std::runtime_error(
                "List of times provided"
                " does not match the number of images!");
    imageTimes = times;
    setTimeOrigin();
}

void ImageStack::resetImages() { images = std::vector<LayeredImage>(); }

void ImageStack::convolvePSF() {
    for (auto& i : images) i.convolvePSF();
}

void ImageStack::saveGlobalMask(const std::string& path) {
    globalMask.saveToFile(path, false);
}

void ImageStack::saveImages(const std::string& path) {
    for (auto& i : images) i.saveLayers(path);
}

const RawImage& ImageStack::getGlobalMask() const { return globalMask; }

std::vector<RawImage> ImageStack::getSciences() {
    std::vector<RawImage> imgs;
    for (auto i : images) imgs.push_back(i.getScience());
    return imgs;
}

std::vector<RawImage> ImageStack::getMasks() {
    std::vector<RawImage> imgs;
    for (auto i : images) imgs.push_back(i.getMask());
    return imgs;
}

std::vector<RawImage> ImageStack::getVariances() {
    std::vector<RawImage> imgs;
    for (auto i : images) imgs.push_back(i.getVariance());
    return imgs;
}

void ImageStack::applyMaskFlags(int flags, const std::vector<int>& exceptions) {
    for (auto& i : images) {
        i.applyMaskFlags(flags, exceptions);
    }
}

void ImageStack::applyGlobalMask(int flags, int threshold) {
    createGlobalMask(flags, threshold);
    for (auto& i : images) {
        i.applyGlobalMask(globalMask);
    }
}

void ImageStack::applyMaskThreshold(float thresh) {
    for (auto& i : images) i.applyMaskThreshold(thresh);
}

void ImageStack::growMask(int steps, bool on_gpu) {
    for (auto& i : images) i.growMask(steps, on_gpu);
}

void ImageStack::createGlobalMask(int flags, int threshold) {
    int ppi = getPPI();

    // For each pixel count the number of images where it is masked.
    std::vector<int> counts(ppi, 0);
    for (unsigned int img = 0; img < images.size(); ++img) {
        float* imgMask = images[img].getMDataRef();
        // Count the number of times a pixel has any of the flags
        for (unsigned int pixel = 0; pixel < ppi; ++pixel) {
            if ((flags & static_cast<int>(imgMask[pixel])) != 0) counts[pixel]++;
        }
    }

    // Set all pixels below threshold to 0 and all above to 1
    float* globalM = globalMask.getDataRef();
    for (unsigned int p = 0; p < ppi; ++p) {
        globalM[p] = counts[p] < threshold ? 0.0 : 1.0;
    }
}

void ImageStack::simpleDifference() {
    RawImage avgTemplate = createAveTemplate();
    for (auto& i : images) i.subtractTemplate(avgTemplate);
}

RawImage ImageStack::createAveTemplate() {
    const int ppi = getPPI();

    // Compute the average value per non-masked pixel.
    std::vector<float> pixel_sum(ppi, 0.0);
    std::vector<float> pixel_count(ppi, 0.0);
    for (auto& i : images) {
        float* img_pix = i.getSDataRef();
        for (unsigned p = 0; p < ppi; ++p) {
            if (img_pix[p] != NO_DATA) {
                pixel_sum[p] += img_pix[p];
                pixel_count[p] += 1.0;
            }
        }
    }
    for (unsigned p = 0; p < ppi; ++p) {
        if (pixel_count[p] > 0.0) {
            pixel_sum[p] = pixel_sum[p] / pixel_count[p];
        } else {
            pixel_sum[p] = 0.0;
        }
    }

    // Build and return the average image.
    RawImage ave_image = RawImage(getWidth(), getHeight(), pixel_sum);
    return ave_image;
}

} /* namespace kbmod */
