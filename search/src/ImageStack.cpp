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
    masterMask = RawImage(getWidth(), getHeight());
    avgTemplate = RawImage(getWidth(), getHeight());
}

ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) {
    verbose = true;
    images = imgs;
    extractImageTimes();
    setTimeOrigin();
    masterMask = RawImage(getWidth(), getHeight());
    avgTemplate = RawImage(getWidth(), getHeight());
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

void ImageStack::saveMasterMask(const std::string& path) {
    // std::cout << masterMask.getWidth() << "\n";
    // std::cout << masterMask.getHeight() << "\n";
    masterMask.saveToFile(path);
    // RawImage test(100, 100);
    // test.saveToFile(path);
}

void ImageStack::saveImages(const std::string& path) {
    for (auto& i : images) i.saveLayers(path);
}

const RawImage& ImageStack::getMasterMask() const { return masterMask; }

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

void ImageStack::applyMasterMask(int flags, int threshold) {
    createMasterMask(flags, threshold);
    for (auto& i : images) {
        i.applyMasterMask(masterMask);
    }
}

void ImageStack::applyMaskThreshold(float thresh) {
    for (auto& i : images) i.applyMaskThreshold(thresh);
}

void ImageStack::growMask(int steps) {
    for (auto& i : images) i.growMask(steps);
}

void ImageStack::createMasterMask(int flags, int threshold) {
    int ppi = getPPI();

    // Initialize masterMask to 0.0s
    float* masterM = masterMask.getDataRef();
    for (unsigned int img = 0; img < images.size(); ++img) {
        float* imgMask = images[img].getMDataRef();
        // Count the number of times a pixel has any of the flags
        for (unsigned int pixel = 0; pixel < ppi; ++pixel) {
            if ((flags & static_cast<int>(imgMask[pixel])) != 0) masterM[pixel]++;
        }
    }

    // Set all pixels below threshold to 0 and all above to 1
    float fThreshold = static_cast<float>(threshold);
    for (unsigned int p = 0; p < ppi; ++p) {
        masterM[p] = masterM[p] < fThreshold ? 0.0 : 1.0;
    }
}

void ImageStack::simpleDifference() {
    createTemplate();
    for (auto& i : images) i.subtractTemplate(avgTemplate);
}

void ImageStack::createTemplate() {
    int ppi = getPPI();
    assert(avgTemplate.getWidth() == getWidth() && avgTemplate.getHeight() == getHeight());
    float* templatePix = avgTemplate.getDataRef();
    for (auto& i : images) {
        float* imgPix = i.getSDataRef();
        for (unsigned p = 0; p < ppi; ++p) templatePix[p] += imgPix[p];
    }

    for (unsigned p = 0; p < ppi; ++p) templatePix[p] /= static_cast<float>(imgCount());
}

} /* namespace kbmod */
