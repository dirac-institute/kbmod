/*
 * PooledImage.h
 *
 * Created on: July 14, 2022
 *
 * A series of RawImages that have been (min or max) pooled to
 * different levels.
 */

#ifndef POOLEDIMAGE_H_
#define POOLEDIMAGE_H_

#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <stdexcept>
#include "RawImage.h"

namespace kbmod {

class PooledImage {
public:
    PooledImage(const RawImage& org_image, int mode);

    // Simple getters.
    unsigned numLevels() const { return images.size(); }
    unsigned getBaseWidth() const { return images[0].getWidth(); }
    unsigned getBaseHeight() const { return images[0].getHeight(); }
    unsigned getBasePPI() const { return images[0].getPPI(); }
    const std::vector<RawImage>& getImages() const { return images; }
    RawImage& getImage(int level) { return images[level]; }

    // Returns the value of a pixel at a given depth where (x, y) is the
    // coordinates on the image at that depth.
    float getPixel(int depth, int x, int y) const;
    
    // Returns the value of a pixel at a given depth where (x, y) is the
    // coordinates on the original image.
    float getMappedPixelAtDepth(int depth, int x, int y) const;

    // Repools an area of +/- width around (x, y) accounting for masking.
    void repoolArea(float x, float y, float radius);
    
    virtual ~PooledImage() {};
    
private:
    std::vector<RawImage> images;
    int pool_mode;
};
    
std::vector<PooledImage> PoolMultipleImages(const std::vector<RawImage>& imagesToPool, int mode);

} /* namespace kbmod */

#endif /* POOLEDIMAGE_H_ */
