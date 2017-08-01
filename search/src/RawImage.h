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
//#include "../pybinds/pybind11/build/mock_install/include/pybind11/pybind11.h"
//#include "../pybinds/pybind11/build/mock_install/include/pybind11/numpy.h"
//#include "../pybinds/pybind11/build/mock_install/include/pybind11/stl.h"
#include "ImageBase.h"
#include "common.h"

namespace kbmod {

extern "C" void
deviceConvolve(float *sourceImg, float *resultImg,
			   int width, int height, PointSpreadFunc *PSF);

class RawImage : public ImageBase {
public:
	RawImage();
	RawImage(unsigned w, unsigned h);
	RawImage(unsigned w, unsigned h, std::vector<float> pix);
	std::vector<float> getPixels();
	float* getDataRef(); // Get pointer to pixels
	void applyMask(int flags, RawImage mask);
	void setAllPix(float value);
	void setPixel(int x, int y, float value);
	void addToPixel(int x, int y, float value);
	//pybind11::array_t<float> toNumpy();
	void saveToFile(std::string path);
	void saveToExtension(std::string path);
	virtual void convolve(PointSpreadFunc psf) override;
	unsigned getWidth() override { return width; }
	unsigned getHeight() override { return height; }
	long* getDimensions() override { return &dimensions[0]; }
	unsigned getPPI() override { return pixelsPerImage; }
	virtual ~RawImage() {};

private:
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
