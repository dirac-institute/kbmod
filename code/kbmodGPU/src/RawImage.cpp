/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

namespace kbmod {

RawImage::RawImage()
{

}



void RawImage::writeFitsImg(std::string path, void *array,
		long *dimensions, unsigned pixelsPerImage)
{
	int status = 0;
	fitsfile *f;
    /* Create file with name */
	fits_create_file(&f, (path).c_str(), &status);

	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, array, &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void RawImage::applyMaskFlags(int flags)
{
	loadLayers();
	mask(flags, &sciencePixels, &maskPixels);
	mask(flags, &variancePixels, &maskPixels);
}

/* Mask all pixels that are not 0 in master mask */
void RawImage::applyMasterMask(std::vector<float> *maskPix)
{
	loadLayers();
	mask(0xFFFFFF, &sciencePixels, maskPix);
	mask(0xFFFFFF, &variancePixels, maskPix);
}

void RawImage::mask(int flags, std::vector<float> *target, std::vector<float> *maskPix)
{
	assert(target->size() == maskPix->size());
	for (unsigned int p=0; p<target->size(); ++p)
	{
		if ((flags & static_cast<int>((*maskPix)[p])) != 0)
			(*target)[p] = MASK_FLAG;
	}
}

void RawImage::setPixel(int x, int y, float value)
{
	sciencePixels[y*width+x] = value;
}

void RawImage::setAllPix(float value)
{
	for (auto& p : sciencePixels) p = value;
}

void RawImage::saveToFile(std::string path) {
	loadLayers();
	writeFitsImg((path+fileName+"SCI.fits"), pixels.data(),
			&dimensions[0], pixelsPerImage);
}

float* RawImage::getDataRef() {
	loadLayers();
	return sciencePixels.data();
}

bool RawImage::isLoaded()
{
	return loaded;
}

double RawImage::getTime()
{
	return captureTime;
}

float RawImage::getWidth()
{
	return width;
}

float RawImage::getHeight()
{
	return height;
}

RawImage::~RawImage() {}

} /* namespace kbmod */
