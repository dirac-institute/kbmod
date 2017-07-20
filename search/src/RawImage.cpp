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
	initDimensions(0,0);
	pixels = std::vector<float>();
}

RawImage::RawImage(unsigned w, unsigned h)
{
	initDimensions(w,h);
	pixels = std::vector<float>(pixelsPerImage);
}

RawImage::RawImage(unsigned w, unsigned h, std::vector<float> pix)
{
	assert(w*h == pix.size());
	initDimensions(w,h);
	pixels = pix;
}

void RawImage::initDimensions(unsigned w, unsigned h)
{
	width = w;
	height = h;
	dimensions[0] = w;
	dimensions[1] = h;
	pixelsPerImage = w*h;
}

void RawImage::writeFitsImg(std::string path)
{
	int status = 0;
	fitsfile *f;

	/* Try opening file */
	if ( fits_open_file(&f, path.c_str(), READWRITE, &status) )
	{
	    /* If no file exists, create file with name */
		fits_create_file(&f, (path).c_str(), &status);
	}

	// This appends a layer (extension) if the file exists)
	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, pixels.data(), &status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void RawImage::convolve(PointSpreadFunc psf)
{
	deviceConvolve(pixels.data(), pixels.data(), getWidth(), getHeight(), &psf);
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
	if (x>=0 && x<width && y>=0 && y<height)
		pixels[y*width+x] = value;
}

void RawImage::addToPixel(int x, int y, float value)
{
	if (x>=0 && x<width && y>=0 && y<height)
		pixels[y*width+x] += value;
}

void RawImage::setAllPix(float value)
{
	for (auto& p : pixels) p = value;
}

/*
pybind11::array_t<float> toNumpy()
{
	return pybind11::array_t<float>(pixels.data(), getPPI());
}
*/

void RawImage::saveToFile(std::string path) {
	writeFitsImg(path);
}

float* RawImage::getDataRef() {
	return pixels.data();
}

} /* namespace kbmod */
