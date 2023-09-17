/*
 * RawImage.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 *
 * RawImage stores pixel level data for a single image.
 */

#ifndef RAWIMAGE_H_
#define RAWIMAGE_H_

#include <algorithm>
#include <array>
#include <vector>
#include <fitsio.h>
#include <float.h>
#include <iostream>
#include <string>
#include <assert.h>
#include <stdexcept>
#ifdef Py_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif
#include "common.h"
#include "PointSpreadFunc.h"

namespace search {

class RawImage {
public:
    RawImage();
    RawImage(const RawImage& old);  // Copy constructor
    RawImage(RawImage&& source);    // Move constructor
    explicit RawImage(unsigned w, unsigned h);
    explicit RawImage(unsigned w, unsigned h, const std::vector<float>& pix);
#ifdef Py_PYTHON_H
    explicit RawImage(pybind11::array_t<float> arr);
    void setArray(pybind11::array_t<float>& arr);
#endif

    // Load the image data from a specific layer of a FITS file.
    void RawImage(const std::string& filePath, int layer_num);

    RawImage& operator=(const RawImage& source);  // Copy assignment
    RawImage& operator=(RawImage&& source);       // Move assignment

    // Basic getter functions for image data.
    unsigned getWidth() const { return width; }
    unsigned getHeight() const { return height; }
    unsigned getNPixels() const { return width * height; }

    // Inline pixel functions.
    float getPixel(int x, int y) const {
        return (x >= 0 && x < width && y >= 0 && y < height) ? pixels[y * width + x] : NO_DATA;
    }

    bool pixelHasData(int x, int y) const {
        return (x >= 0 && x < width && y >= 0 && y < height) ? pixels[y * width + x] != NO_DATA : false;
    }

    void setPixel(int x, int y, float value) {
        if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] = value;
    }
    const std::vector<float>& getPixels() const { return pixels; }
    float* getDataRef() { return pixels.data(); }  // Get pointer to pixels

    // Get the interpolated brightness of a real values point
    // using the four neighboring pixels.
    float getPixelInterp(float x, float y) const;

    // Check if two raw images are approximately equal.
    bool approxEqual(const RawImage& imgB, float atol) const;

    // Functions for locally storing the image time.
    float getObstime() const;
    void setObstime(float new_time) { obstime = new_time; }

    // Compute the min and max bounds of values in the image.
    std::array<float, 2> computeBounds() const;

    // Masks out the pixels of the image where:
    //   flags a bit vector of mask flags to apply
    //       (use 0xFFFFFF to apply all flags)
    //   exceptions is a vector of pixel flags to ignore
    //   mask is an image of bit vector mask flags
    void applyMask(int flags, const std::vector<int>& exceptions, const RawImage& mask);

    void setAllPix(float value);
    void addToPixel(float fx, float fy, float value);
    void addPixelInterp(float x, float y, float value);
    std::vector<float> bilinearInterp(float x, float y) const;

    // Grow the area of masked pixels.
    void growMask(int steps);

    // Save the RawImage to a file. Append indicates whether to append
    // or create a new file.
    void saveToFile(const std::string& path, bool append);

    // Convolve the image with a point spread function.
    void convolve(PointSpreadFunc psf);
    void convolve_cpu(const PointSpreadFunc& psf);

    // Create a "stamp" image of a give radius (width=2*radius+1)
    // about the given point.
    // keep_no_data indicates whether to use the NO_DATA flag or replace with 0.0.
    RawImage createStamp(float x, float y, int radius, bool interpolate, bool keep_no_data) const;

    // The maximum value of the image and return the coordinates. The parameter
    // furthest_from_center indicates whether to break ties using the peak further
    // or closer to the center of the image.
    pixelPos findPeak(bool furthest_from_center) const;

    // Find the basic image moments in order to test if stamps have a gaussian shape.
    // It computes the moments on the "normalized" image where the minimum
    // value has been shifted to zero and the sum of all elements is 1.0.
    // Elements with NO_DATA are treated as zero.
    imageMoments findCentralMoments() const;

    virtual ~RawImage(){};

private:
    unsigned width;
    unsigned height;
    std::vector<float> pixels;
    float obstime;
};

// Helper functions for creating composite images.
RawImage createMedianImage(const std::vector<RawImage>& images);
RawImage createSummedImage(const std::vector<RawImage>& images);
RawImage createMeanImage(const std::vector<RawImage>& images);

} /* namespace search */

#endif /* RAWIMAGE_H_ */
