/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

RawImage::RawImage(std::string path) {

	filePath = path;
	readHeader();

	// Buffers to hold the 3 image layers read by cfitsio
	float *sBuffer = new float[pixelsPerImage];
	float *vBuffer = new float[pixelsPerImage];
	float *mBuffer = new float[pixelsPerImage];

	// Load images from file
	readFitsImg((filePath+"[1]").c_str(), sBuffer);
	readFitsImg((filePath+"[2]").c_str(), vBuffer);
	readFitsImg((filePath+"[3]").c_str(), mBuffer);

	sciencePixels = std::vector<float>(sBuffer, sBuffer+pixelsPerImage);
	variancePixels = std::vector<float>(vBuffer, vBuffer+pixelsPerImage);
	maskPixels = std::vector<float>(mBuffer, mBuffer+pixelsPerImage);

	delete sBuffer;
	delete vBuffer;
	delete mBuffer;
}

/* Read the image dimensions and capture time from header */
void RawImage::readHeader()
{
	fitsfile *fptr;
	int status = 0;
	int fileNotFound;

	// Open header to read MJD
	if (fits_open_file(&fptr, filePath.c_str(), READONLY, &status))
		fits_report_error(stderr, status);

	// Read image capture time
	if (fits_read_key(fptr, TFLOAT, "MJD", &captureTime, NULL, &status))
		fits_report_error(stderr, status);

	if (fits_close_file(fptr, &status))
		fits_report_error(stderr, status);

	// Reopen header for first layer to get image dimensions
	if (fits_open_file(&fptr, (filePath+"[1]").c_str(), READONLY, &status))
		fits_report_error(stderr, status);

	// Read image Dimensions
	if (fits_read_keys_lng(fptr, "NAXIS", 1, 2, dimensions, &fileNotFound, &status))
		fits_report_error(stderr, status);

	width = dimensions[0];
	height = dimensions[1];
	// Calculate pixels per image from dimensions x*y
	pixelsPerImage = dimensions[0]*dimensions[1];

	if (fits_close_file(fptr, &status))
		fits_report_error(stderr, status);
}

void RawImage::readFitsImg(const char *name, float *target)
{
	fitsfile *fptr;
	int nullval = 0;
	int anynull;
	int status = 0;

	if (fits_open_file(&fptr, name, READONLY, &status))
		fits_report_error(stderr, status);
	if (fits_read_img(fptr, TFLOAT, 1, pixelsPerImage,
		&nullval, target, &anynull, &status))
		fits_report_error(stderr, status);
	if (fits_close_file(fptr, &status))
		fits_report_error(stderr, status);
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

float RawImage::getTime()
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
