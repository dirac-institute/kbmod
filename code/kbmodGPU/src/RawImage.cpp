/*
 * RawImage.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "RawImage.h"

RawImage::RawImage(std::string path)
{
	filePath = path;
	int fBegin = path.find_last_of("/");
	int fEnd = path.find_last_of(".fits")-4;
	fileName = path.substr(fBegin, fEnd-fBegin);
	readHeader();
	layersLoaded = false;
}

void RawImage::loadLayers()
{
	if (!layersLoaded)
	{
		// Buffers to hold the 3 image layers read by cfitsio
		float *sBuffer = new float[pixelsPerImage];
		float *mBuffer = new float[pixelsPerImage];
		float *vBuffer = new float[pixelsPerImage];

		// Load images from file
		readFitsImg((filePath+"[1]").c_str(), sBuffer);
		readFitsImg((filePath+"[2]").c_str(), mBuffer);
		readFitsImg((filePath+"[3]").c_str(), vBuffer);

		sciencePixels = std::vector<float>(sBuffer, sBuffer+pixelsPerImage);
		maskPixels = std::vector<float>(mBuffer, mBuffer+pixelsPerImage);
		variancePixels = std::vector<float>(vBuffer, vBuffer+pixelsPerImage);

		delete sBuffer;
		delete mBuffer;
		delete vBuffer;
		layersLoaded = true;

	}
}

void RawImage::freeLayers()
{
	if (layersLoaded)
	{
		sciencePixels = std::vector<float>();
		maskPixels = std::vector<float>();
		variancePixels = std::vector<float>();
		layersLoaded = false;
	}
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

void RawImage::saveSci(std::string path) {
	loadLayers();
	writeFitsImg((path+fileName+"SCI.fits"), sciencePixels.data(),
			&dimensions[0], pixelsPerImage);
}
void RawImage::saveMask(std::string path) {
	loadLayers();
	writeFitsImg((path+fileName+"MASK.fits"), maskPixels.data(),
			&dimensions[0], pixelsPerImage);
}
void RawImage::saveVar(std::string path){
	loadLayers();
	writeFitsImg((path+fileName+"VAR.fits"), variancePixels.data(),
			&dimensions[0], pixelsPerImage);
}

float* RawImage::getSDataRef() {
	loadLayers();
	return sciencePixels.data();
}

float* RawImage::getVDataRef() {
	loadLayers();
	return variancePixels.data();
}

float* RawImage::getMDataRef() {
	loadLayers();
	return maskPixels.data();
}

bool RawImage::isLoaded()
{
	return layersLoaded;
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
