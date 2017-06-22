/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

RawImage::RawImage(float *data, int x , int y) {
	pixels = std::vector<float>(data, data+x*y);
	width =  x;
	height = y;
}

float* RawImage::getPixelsRef() {
	return pixels.data();
}

RawImage::~RawImage() {}

