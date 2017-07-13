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
	std::vector<float> empty(0);
	setData(0,0, empty.data());
}

RawImage::RawImage(unsigned w, unsigned h, float *pix)
{
	setData(w,h,pix);
}

void RawImage::setData(unsigned w, unsigned h, float *pix)
{
	width = w;
	height = h;
	dimensions[0] = w;
	dimensions[1] = h;
	pixelsPerImage = w*h;
	pixels.assign(pix, pix+pixelsPerImage);
}


void RawImage::writeFitsImg(std::string path)
{
	int status = 0;
	fitsfile *f;
    /* Create file with name */
	fits_create_file(&f, (path).c_str(), &status);

	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, pixels.data(), &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void RawImage::applyMask(int flags, RawImage mask)
{
	float *maskPix = mask.getDataRef();
	assert(pixelsPerImage == mask.getPPI());
	for (unsigned int p=0; p<pixelsPerImage; ++p)
	{
		if ((flags & static_cast<int>(maskPix[p])) != 0)
			pixels[p] = MASK_FLAG;
	}
}

void RawImage::setPixel(int x, int y, float value)
{
	pixels[y*width+x] = value;
}

void RawImage::setAllPix(float value)
{
	for (auto& p : pixels) p = value;
}

void RawImage::saveToFile(std::string path) {
	writeFitsImg(path);
}

float* RawImage::getDataRef() {
	return pixels.data();
}

} /* namespace kbmod */
