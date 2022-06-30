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
	unsigned getWidth() override { return width; }
	unsigned getHeight() override { return height; }
	long* getDimensions() override { return &dimensions[0]; }
	unsigned getPPI() override { return pixelsPerImage; }
	float getPixel(int x, int y);
	float getPixelInterp(float x, float y);
	bool pixelHasData(int x, int y);
	std::vector<float> getPixels();
	float* getDataRef(); // Get pointer to pixels

	// Masks out the pixels of the image where:
	//   flags a bit vector of mask flags to apply 
	//       (use 0xFFFFFF to apply all flags)
	//   exceptions is a vector of pixel flags to ignore
	//   mask is an image of bit vector mask flags
	void applyMask(int flags, std::vector<int> exceptions,
	               RawImage mask);

	void setAllPix(float value);
	void setPixel(int x, int y, float value);
	void addToPixel(float fx, float fy, float value);
	void addPixelInterp(float x, float y, float value);

	// Mask out an object 
	void maskObject(float x, float y, PointSpreadFunc psf);
	void maskPixelInterp(float x, float y);
	void growMask();
	std::vector<float> bilinearInterp(float x, float y);

	// Save the RawImage to a file.
	void saveToFile(std::string path);
	void saveToExtension(std::string path);

	// Convolve the image with a point spread function.
	virtual void convolve(PointSpreadFunc psf) override;

	// Creates images of half the height and width where each
	// pixel  is either the min or max (depending on mode) of 
	// the local pixels in the original image.
	RawImage pool(short mode);
	RawImage poolMin() { return pool(POOL_MIN); }
	RawImage poolMax() { return pool(POOL_MAX); }

	virtual ~RawImage() {};

private:
	float pixelOverlap(float px, float py, float x, float y);
	void initDimensions(unsigned w, unsigned h);
	void writeFitsImg(std::string path);
	void writeFitsExtension(std::string path);
	unsigned width;
	unsigned height;
	long dimensions[2];
	unsigned pixelsPerImage;
	std::vector<float> pixels;
};

} /* namespace kbmod */

#endif /* RAWIMAGE_H_ */
