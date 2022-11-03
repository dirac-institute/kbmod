/*
 * ImageStack.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 * ImageStack stores a series of LayeredImages from different times.
 */

#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <vector>
#include <dirent.h>
#include <string>
#include <list>
#include <iostream>
#include <stdexcept>
#include "LayeredImage.h"

namespace search {

class ImageStack {
public:
    ImageStack(const std::vector<std::string>& filenames, const std::vector<PointSpreadFunc>& psfs);
    ImageStack(const std::vector<LayeredImage>& imgs);

    // Simple getters.
    unsigned imgCount() const;
    unsigned getWidth() const { return images[0].getWidth(); }
    unsigned getHeight() const { return images[0].getHeight(); }
    unsigned getPPI() const { return images[0].getPPI(); }
    const std::vector<float>& getTimes() const;
    float* getTimesDataRef();
    LayeredImage& getSingleImage(int index);

    // Simple setters.
    void setTimes(const std::vector<float>& times);
    void resetImages();
    void setSingleImage(int index, LayeredImage& img);

    // Get a vector of images or layers.
    std::vector<LayeredImage>& getImages();
    std::vector<RawImage> getSciences();
    std::vector<RawImage> getMasks();
    std::vector<RawImage> getVariances();

    // Apply makes to all the images.
    void applyGlobalMask(int flags, int threshold);
    void applyMaskFlags(int flags, const std::vector<int>& exceptions);
    void applyMaskThreshold(float thresh);
    void growMask(int steps, bool on_gpu);
    const RawImage& getGlobalMask() const;

    void convolvePSF();
    void simpleDifference();

    // Save data to files.
    void saveGlobalMask(const std::string& path);
    void saveImages(const std::string& path);

    virtual ~ImageStack(){};

private:
    void loadImages(const std::vector<std::string>& fileNames, const std::vector<PointSpreadFunc>& psfs);
    void extractImageTimes();
    void setTimeOrigin();
    void createGlobalMask(int flags, int threshold);
    RawImage createAveTemplate();
    std::vector<LayeredImage> images;
    RawImage globalMask;
    std::vector<float> imageTimes;
    bool verbose;
};

} /* namespace search */

#endif /* IMAGESTACK_H_ */
