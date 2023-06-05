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
    unsigned imgCount() const { return images.size(); }
    unsigned getWidth() const { return images.size() > 0 ? images[0].getWidth() : 0; }
    unsigned getHeight() const { return images.size() > 0 ? images[0].getHeight() : 0; }
    unsigned getPPI() const { return images.size() > 0 ? images[0].getPPI() : 0; }
    std::vector<LayeredImage>& getImages() { return images; }
    const std::vector<float>& getTimes() const { return imageTimes; }
    float* getTimesDataRef() { return imageTimes.data(); }
    LayeredImage& getSingleImage(int index);

    // Simple setters.
    void setTimes(const std::vector<float>& times);
    void resetImages();
    void setSingleImage(int index, LayeredImage& img);

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

    // Create a RawImage from the shift-stacked images.
    RawImage simpleShiftAndStack(float v_x, float v_y, bool use_mean);

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
