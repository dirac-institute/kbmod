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

#include <vector>
#include <fitsio.h>
#include <iostream>
#include <string>
#include <assert.h>
#include <stdexcept>
#ifdef Py_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif
#include "ImageBase.h"
#include "common.h"

namespace kbmod {

// Performs convolution between an image represented as an array of floats
// and a PSF on a GPU device.
extern "C" void
deviceConvolve(float *sourceImg, float *resultImg,
	int width, int height, float *psfKernel,
	int psfSize, int psfDim, int psfRadius, float psfSum);

// Performs pixel pooling on an image represented as an array of floats.
// on a GPU device.
extern "C" void
devicePool(int sourceWidth, int sourceHeight, float *source,
	int destWidth, int destHeight, float *dest, char mode);

class RawImage : public ImageBase {
public:
	RawImage();
	RawImage(unsigned w, unsigned h);
	RawImage(unsigned w, unsigned h, std::vector<float> pix);
#ifdef Py_PYTHON_H
	RawImage(pybind11::array_t<float> arr);
	void setArray(pybind11::array_t<float>& arr);
#endif

	// Basic getter functions for image data.
	unsigned getWidth() const override { return width; }
	unsigned getHeight() const override { return height; }
	long* getDimensions() override { return &dimensions[0]; }
	unsigned getPPI() const override { return pixelsPerImage; }
	float getPixel(int x, int y) const;
	bool pixelHasData(int x, int y) const;
	const std::vector<float>& getPixels() const;
	float* getDataRef(); // Get pointer to pixels

	// Get the interpolated brightness of a real values point
	// using the four neighboring pixels.
	float getPixelInterp(float x, float y) const;
  
	// Masks out the pixels of the image where:
	//   flags a bit vector of mask flags to apply 
	//       (use 0xFFFFFF to apply all flags)
	//   exceptions is a vector of pixel flags to ignore
	//   mask is an image of bit vector mask flags
	void applyMask(int flags, const std::vector<int>& exceptions,
	               const RawImage& mask);

	void setAllPix(float value);
	void setPixel(int x, int y, float value);
	void addToPixel(float fx, float fy, float value);
	void addPixelInterp(float x, float y, float value);

	// Mask out an object 
	void maskObject(float x, float y, const PointSpreadFunc& psf);
	void maskPixelInterp(float x, float y);
	void growMask(int steps);
	std::vector<float> bilinearInterp(float x, float y) const;

	// Save the RawImage to a file.
	void saveToFile(const std::string& path);
	void saveToExtension(const std::string& path);

	// Convolve the image with a point spread function.
	virtual void convolve(PointSpreadFunc psf) override;

	// Create a "stamp" image of a give radius (width=2*radius+1)
	// about the given point.
	RawImage createStamp(float x, float y, int radius, bool interpolate) const;

	// Creates images of half the height and width where each
	// pixel is either the min or max (depending on mode) of
	// the local pixels in the original image.
	RawImage pool(short mode);
	RawImage poolMin() { return pool(POOL_MIN); }
	RawImage poolMax() { return pool(POOL_MAX); }

	// Compute the pooling function over an arbitrary region.
	// lx <= x <= hx and ly <= y <= hy.
	float extremeInRegion(int lx, int ly, int hx, int hy, short pool_mode);

	virtual ~RawImage() {};

private:
	float pixelOverlap(float px, float py, float x, float y);
	void initDimensions(unsigned w, unsigned h);
	void writeFitsImg(const std::string& path);
	void writeFitsExtension(const std::string& path);
	unsigned width;
	unsigned height;
	long dimensions[2];
	unsigned pixelsPerImage;
	std::vector<float> pixels;
};

// Helper functions for creating composite images.
RawImage createMedianImage(const std::vector<RawImage>& images);
RawImage createSummedImage(const std::vector<RawImage>& images);
RawImage createMeanImage(const std::vector<RawImage>& images);

} /* namespace kbmod */

#endif /* RAWIMAGE_H_ */
