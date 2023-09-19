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

#include "common.h"
#include "LayeredImage.h"
#include "RawImage.h"

namespace image_base {

class ImageStack {
public:
    ImageStack(const std::vector<std::string>& filenames, const std::vector<PointSpreadFunc>& psfs);
    ImageStack(const std::vector<LayeredImage>& imgs);

    // Simple getters.
    unsigned imgCount() const { return images.size(); }
    unsigned getWidth() const { return images.size() > 0 ? images[0].getWidth() : 0; }
    unsigned getHeight() const { return images.size() > 0 ? images[0].getHeight() : 0; }
    unsigned getNPixels() const { return images.size() > 0 ? images[0].getNPixels() : 0; }
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
    void growMask(int steps);
    const RawImage& getGlobalMask() const;

    void convolvePSF();

    // Save data to files.
    void saveGlobalMask(const std::string& path);
    void saveImages(const std::string& path);

    virtual ~ImageStack(){};

private:
    void loadImages(const std::vector<std::string>& fileNames, const std::vector<PointSpreadFunc>& psfs);
    void extractImageTimes();
    void setTimeOrigin();
    void createGlobalMask(int flags, int threshold);
    std::vector<LayeredImage> images;
    RawImage globalMask;
    std::vector<float> imageTimes;
    bool verbose;
};

} /* namespace image_base */

#endif /* IMAGESTACK_H_ */
