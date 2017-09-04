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

RawImage RawImage::pool(short mode)
{
	// Half the dimensions, rounded up
    int pooledWidth = (getWidth()+1)/2;
    int pooledHeight = (getHeight()+1)/2;
	RawImage pooledImage = RawImage(pooledWidth, pooledHeight);
	devicePool(getWidth(), getHeight(), pixels.data(),
			      pooledWidth, pooledHeight, pooledImage.getDataRef(), mode);
	return pooledImage;
}

void RawImage::applyMask(int flags, std::vector<int> exceptions, RawImage mask)
{
	float *maskPix = mask.getDataRef();
	assert(pixelsPerImage == mask.getPPI());
	for (unsigned int p=0; p<pixelsPerImage; ++p)
	{
		int pixFlags = static_cast<int>(maskPix[p]);
		bool isException = false;
		for (auto& e : exceptions)
			isException = isException || e == pixFlags;
		if ( !isException && ((flags & pixFlags ) != 0))
			pixels[p] = NO_DATA;
	}
}

std::vector<float> RawImage::bilinearInterp(float x, float y)
{
	// Linearl interpolation
	// Find the 4 pixels (aPix, bPix, cPix, dPix)
	// that the corners (a, b, c, d) of the
	// new pixel land in, and blend into those

	// Returns a vector with 4 pixel locations
	// and their interpolation value

	// Top right
	float ax = x + 0.5;
	float ay = y + 0.5;
	float aPx = floor(ax);
	float aPy = floor(ay);
	float aAmount = (ax-aPx)*(ay-aPy);

	// Bottom right
	float bx = x + 0.5;
	float by = y - 0.5;
	float bPx = floor(bx);
	float bPy = floor(by);
	float bAmount = (bx-bPx)*(bPy+1.0-by);

	// Bottom left
	float cx = x - 0.5;
	float cy = y - 0.5;
	float cPx = floor(cx);
	float cPy = floor(cy);
	float cAmount = (cPx+1.0-cx)*(cPy+1.0-cy);

	// Top left
	float dx = x - 0.5;
	float dy = y + 0.5;
	float dPx = floor(dx);
	float dPy = floor(dy);
	float dAmount = (dPx+1.0-dx)*(dy-dPy);

	// make sure the right amount has been distributed
	assert(std::abs(aAmount+bAmount+cAmount+dAmount-1.0)<0.0001);
	return { aPx, aPy, aAmount,
		     bPx, bPy, bAmount,
		     cPx, cPy, cAmount,
		     dPx, dPy, dAmount };
}

void RawImage::addPixelInterp(float x, float y, float value)
{
	// Interpolation values
	std::vector<float> iv = bilinearInterp(x,y);

	addToPixel(iv[0], iv[1], value*iv[2]);

	addToPixel(iv[3], iv[4], value*iv[5]);

	addToPixel(iv[6], iv[7], value*iv[8]);

	addToPixel(iv[9], iv[10],value*iv[11]);
}

void RawImage::addToPixel(float fx, float fy, float value)
{
	int x = static_cast<int>(fx);
	int y = static_cast<int>(fy);
	if (x>=0 && x<width && y>=0 && y<height)
		pixels[y*width+x] += value;
}

void RawImage::setPixel(int x, int y, float value)
{
	if (x>=0 && x<width && y>=0 && y<height)
		pixels[y*width+x] = value;
}

float RawImage::getPixel(int x, int y)
{
	if (x>=0 && x<width && y>=0 && y<height) {
		return pixels[y*width+x];
	} else {
		return NO_DATA;
	}
}

float RawImage::getPixelInterp(float x, float y)
{
	if ((x<0.0 || y<0.0) || (x>static_cast<float>(width) ||
	     y>static_cast<float>(height))) return NO_DATA;
	std::vector<float> iv = bilinearInterp(x,y);
	float a = getPixel(iv[0], iv[1]);
	float b = getPixel(iv[3], iv[4]);
	float c = getPixel(iv[6], iv[7]);
	float d = getPixel(iv[9], iv[10]);
	float interpSum = 0.0;
	float total = 0.0;
	if (a != NO_DATA) {
		interpSum += iv[2];
		total += a*iv[2];
	}
	if (b != NO_DATA) {
		interpSum += iv[5];
		total += b*iv[5];
	}
	if (c != NO_DATA) {
		interpSum += iv[8];
		total += c*iv[8];
	}
	if (d != NO_DATA) {
		interpSum += iv[11];
		total += d*iv[11];
	}
	assert(interpSum != 0.0);
	return total/interpSum;
}

void RawImage::setAllPix(float value)
{
	for (auto& p : pixels) p = value;
}

float* RawImage::getDataRef() {
	return pixels.data();
}

} /* namespace kbmod */
