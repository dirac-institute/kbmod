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
	science = RawImage(width, height);
	mask =  RawImage(width, height);
	variance = RawImage(width, height);
	loadLayers();
}

LayeredImage::LayeredImage(std::string name, int w, int h,
		float noiseStDev, float pixelVariance, double time)
{
	fileName = name;
	pixelsPerImage = w*h;
	width = w;
	height = h;
	captureTime = time;
	std::vector<float> rawSci(pixelsPerImage);
	std::default_random_engine generator;
	std::normal_distribution<float> distrib(0.0, noiseStDev);
	for (float& p : rawSci) p = distrib(generator);
	science = RawImage(w,h, rawSci);
	mask = RawImage(w,h,std::vector<float>(pixelsPerImage, 0.0));
	variance = RawImage(w,h,std::vector<float>(pixelsPerImage, pixelVariance));
}

/* Read the image dimensions and capture time from header */
void LayeredImage::readHeader()
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
	// Load images from file into layers' pixels
	readFitsImg((filePath+"[1]").c_str(), science.getDataRef());
	readFitsImg((filePath+"[2]").c_str(), mask.getDataRef());
	readFitsImg((filePath+"[3]").c_str(), variance.getDataRef());
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

void LayeredImage::addObject(float x, float y, float flux, PointSpreadFunc psf)
{
	std::vector<float> k = psf.getKernel();
	int dim = psf.getDim();
	int initialX = static_cast<int>(x+0.5)-psf.getRadius();
	int initialY = static_cast<int>(y+0.5)-psf.getRadius();
	int count = 0;
	// Does x/y order need to be flipped?
	for (int i=0; i<dim; ++i)
	{
		for (int j=0; j<dim; ++j)
		{
			science.addToPixel(initialX+i, initialY+j, flux*k[count]);
			count++;
		}
	}
}

void LayeredImage::convolve(PointSpreadFunc psf)
{
	PointSpreadFunc psfSQ(psf.getStdev());
	psfSQ.squarePSF();
	science.convolve(psf);
	variance.convolve(psfSQ);
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

/*
pybind11::array_t<float> sciToNumpy()
{
	return science.toNumpy();
}
*/

void LayeredImage::saveLayers(std::string path)
{
	fitsfile *fptr;
	int status = 0;
	long naxes[2] = {0,0};
	fits_create_file(&fptr, (path+fileName+".fits").c_str(), &status);
    fits_create_img(fptr, SHORT_IMG, 0, naxes, &status);
	fits_update_key(fptr, TDOUBLE, "MJD", &captureTime,
			"[d] Generated Image time", &status);
	fits_close_file(fptr, &status);
	fits_report_error(stderr, status);

	science.saveToExtension(path+fileName+".fits");
	mask.saveToExtension(path+fileName+".fits");
	variance.saveToExtension(path+fileName+".fits");
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

} /* namespace kbmod */
