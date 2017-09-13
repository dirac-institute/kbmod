/*
 * RawImage.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
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

extern "C" void
deviceConvolve(float *sourceImg, float *resultImg,
	int width, int height, float *psfKernel,
	int psfSize, int psfDim, int psfRadius, float psfSum);

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
	std::vector<float> getPixels();
	float* getDataRef(); // Get pointer to pixels
	void applyMask(int flags,
			std::vector<int> exceptions, RawImage mask);
	void setAllPix(float value);
	void setPixel(int x, int y, float value);
	void addPixelInterp(float x, float y, float value);
	void maskObject(float x, float y, PointSpreadFunc psf);
	void maskPixelInterp(float x, float y);
	std::vector<float> bilinearInterp(float x, float y);
	float getPixel(int x, int y);
	float getPixelInterp(float x, float y);
	//pybind11::array_t<float> toNumpy();
	void saveToFile(std::string path);
	void saveToExtension(std::string path);
	virtual void convolve(PointSpreadFunc psf) override;
	RawImage pool(short mode);
	RawImage poolMin() { return pool(POOL_MIN); }
	RawImage poolMax() { return pool(POOL_MAX); }
	unsigned getWidth() override { return width; }
	unsigned getHeight() override { return height; }
	long* getDimensions() override { return &dimensions[0]; }
	unsigned getPPI() override { return pixelsPerImage; }
	virtual ~RawImage() {};

private:
	void addToPixel(float fx, float fy, float value);
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
