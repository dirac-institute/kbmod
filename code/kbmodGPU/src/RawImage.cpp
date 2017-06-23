/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

RawImage::RawImage(float *sData, float *vData,
		    	   float *mData, int x, int y, float time) {
	pixelsPerImg = x*y;
	width =  x;
	height = y;
	captureTime = time;
	sciencePixels = std::vector<float>(sData, sData+pixelsPerImg);
	variancePixels = std::vector<float>(vData, vData+pixelsPerImg);
	maskPixels = std::vector<float>(mData, mData+pixelsPerImg);
}

float* RawImage::getSDataRef() {
	return sciencePixels.data();
}

float* RawImage::getVDataRef() {
	return variancePixels.data();
}

float* RawImage::getMDataRef() {
	return maskPixels.data();
}

float getTime()
{
	return captureTime;
}

RawImage::~RawImage() {}
