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

RawImage::RawImage(unsigned w, unsigned h) : pixels(w*h)
{
	initDimensions(w,h);
}

RawImage::RawImage(unsigned w, unsigned h,
		std::vector<float> pix) : pixels(pix)
{
	assert(w*h == pix.size());
	initDimensions(w,h);
	//pixels = pix;
}

void RawImage::initDimensions(unsigned w, unsigned h)
{
	width = w;
	height = h;
	dimensions[0] = w;
	dimensions[1] = h;
	pixelsPerImage = w*h;
}

void RawImage::writeFitsExtension(std::string path)
{
	int status = 0;
	fitsfile *f;

	/* Try opening file */
	if ( fits_open_file(&f, path.c_str(), READWRITE, &status) )
	{
	    /* If no file exists, create file with name */
		fits_create_file(&f, (path).c_str(), &status);
		fits_report_error(stderr, status);
	}

	// This appends a layer (extension) if the file exists)
	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
	fits_report_error(stderr, status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, pixels.data(), &status);
	fits_report_error(stderr, status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void RawImage::writeFitsImg(std::string path)
{
	int status = 0;
	fitsfile *f;

	//fits_open_file(&f, path.c_str(), READWRITE, &status);
	fits_create_file(&f, (path).c_str(), &status);
	fits_report_error(stderr, status);

	// This appends a layer (extension) if the file exists)
	/* Create the primary array image (32-bit float pixels) */
	fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
	fits_report_error(stderr, status);

	/* Write the array of floats to the image */
	fits_write_img(f, TFLOAT, 1, pixelsPerImage, pixels.data(), &status);
	fits_report_error(stderr, status);
	fits_close_file(f, &status);
	fits_report_error(stderr, status);
}

void RawImage::saveToFile(std::string path) {
	writeFitsImg(path);
}

void RawImage::saveToExtension(std::string path) {
	writeFitsExtension(path);
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

void RawImage::addPixelInterp(float x, float y, float value)
{
	// Linearly interpolation
	// Find the 4 pixels (aPix, bPix, cPix, dPix)
	// that the corners (a, b, c, d) of the
	// new pixel land in, and blend into those

	// Top right
	float ax = x + 0.5;
	float ay = y + 0.5;
	float aPx = floor(ax);
	float aPy = floor(ay);
	float aAmount = value*(ax-aPx)*(ay-aPy);
	addToPixel(aPx, aPy, aAmount);

	// Bottom right
	float bx = x + 0.5;
	float by = y - 0.5;
	float bPx = floor(bx);
	float bPy = floor(by);
	float bAmount = value*(bx-bPx)*(bPy+1.0-by);
	addToPixel(bPx, bPy, bAmount);

	// Bottom left
	float cx = x - 0.5;
	float cy = y - 0.5;
	float cPx = floor(cx);
	float cPy = floor(cy);
	float cAmount = value*(cPx+1.0-cx)*(cPy+1.0-cy);
	addToPixel(cPx, cPy, cAmount);

	// Top left
	float dx = x - 0.5;
	float dy = y + 0.5;
	float dPx = floor(dx);
	float dPy = floor(dy);
	float dAmount = value*(dPx+1.0-dx)*(dy-dPy);
	addToPixel(dPx, dPy, dAmount);

}

void RawImage::addToPixel(float fx, float fy, float value)
{
	int x = static_cast<int>(fx);
	int y = static_cast<int>(fy);
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

float* RawImage::getDataRef() {
	return pixels.data();
}

} /* namespace kbmod */
