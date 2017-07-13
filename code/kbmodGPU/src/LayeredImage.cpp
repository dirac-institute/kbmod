/*
 * LayeredImage.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 */

#include "LayeredImage.h"

namespace kbmod {

LayeredImage::LayeredImage(std::string path) {
	filePath = path;
	int fBegin = path.find_last_of("/");
	int fEnd = path.find_last_of(".fits")-4;
	fileName = path.substr(fBegin, fEnd-fBegin);
	readHeader();
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
	if (fits_read_key(fptr, TDOUBLE, "MJD", &captureTime, NULL, &status))
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

void LayeredImage::loadLayers()
{

	// Buffers to hold the 3 image layers read by cfitsio
	float *sBuffer = new float[pixelsPerImage];
	float *mBuffer = new float[pixelsPerImage];
	float *vBuffer = new float[pixelsPerImage];

	// Load images from file
	readFitsImg((filePath+"[1]").c_str(), sBuffer);
	readFitsImg((filePath+"[2]").c_str(), mBuffer);
	readFitsImg((filePath+"[3]").c_str(), vBuffer);

	science(width, height, sBuffer);
	mask(width, height, mBuffer);
	variance(width, height, vBuffer);

	delete sBuffer;
	delete mBuffer;
	delete vBuffer;

}

void LayeredImage::readFitsImg(const char *name, float *target)
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

void LayeredImage::applyMaskFlags(int flags)
{
	science.applyMask(flags, mask);
	variance.applyMask(flags, mask);
}

/* Mask all pixels that are not 0 in master mask */
void LayeredImage::applyMasterMask(RawImage masterM)
{
	science.applyMask(0xFFFFFF, masterM);
	variance.applyMask(0xFFFFFF, masterM);
}

void LayeredImage::saveSci(std::string path) {
	science.saveToFile(path+fileName+"SCI.fits");
}

void LayeredImage::saveMask(std::string path) {
	mask.saveToFile(path+fileName+"MASK.fits");
}

void LayeredImage::saveVar(std::string path){
	variance.saveToFile(path+fileName+"VAR.fits");
}

float* LayeredImage::getSDataRef() {
	return science.getDataRef();
}

float* LayeredImage::getMDataRef() {
	return mask.getDataRef();
}

float* LayeredImage::getVDataRef() {
	return variance.getDataRef();
}

double LayeredImage::getTime()
{
	return captureTime;
}

LayeredImage::~LayeredImage() {
	// TODO Auto-generated destructor stub
}

} /* namespace kbmod */
