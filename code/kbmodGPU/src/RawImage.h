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
#include "common.h"

namespace kbmod {

class RawImage : public ImageBase {
public:
	RawImage(unsigned w, unsigned h, float *pix);
	std::vector<float> getPixels();
	float* getDataRef(); // Get pointer to science pixels
	void applyMask(std::vector<float> *maskPix);
	void setAllPix(float value);
	void setPixel(int x, int y, float value);
	void saveToFile(std::string path);
	virtual ~RawImage();

private:
	void writeFitsImg(std::string path);
	void mask(int flag, std::vector<float> *target, std::vector<float> *maskPix);
	unsigned width;
	unsigned height;
	long dimensions[2];
	unsigned pixelsPerImage;
	std::vector<float> pixels;
};

} /* namespace kbmod */

#endif /* RAWIMAGE_H_ */
