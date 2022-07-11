/*
 * LayeredImage.h
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 *
 *  LayeredImage stores an image from a single time with different layers of 
 *  data, such as science pixels, variance pixels, and mask pixels.
 */

#ifndef LAYEREDIMAGE_H_
#define LAYEREDIMAGE_H_

#include <vector>
#include <fitsio.h>
#include <iostream>
#include <string>
#include <random>
#include <assert.h>
#include <stdexcept>
#include "RawImage.h"
#include "common.h"

namespace kbmod {

class LayeredImage : public ImageBase {
public:
	LayeredImage(std::string path);
	LayeredImage(std::string name, int w, int h,
		float noiseStDev, float pixelVariance, double time);

	// Basic getter functions for image data.
	std::string getName() { return fileName; }
	unsigned getWidth() const override { return width; }
	unsigned getHeight() const override { return height; }
	long* getDimensions() override { return &dimensions[0]; }
	unsigned getPPI() const override { return pixelsPerImage; }
	double getTime() const;

	// Getter functions for the data in the individual layers.
	RawImage& getScience();
	RawImage& getMask();
	RawImage& getVariance();
	float* getSDataRef(); // Get pointer to science pixels
	float* getVDataRef(); // Get pointer to variance pixels
	float* getMDataRef(); // Get pointer to mask pixels

	// Applies the mask functions to each of the science and variance layers. 
	void applyMaskFlags(int flag, const std::vector<int>& exceptions);
	void applyMasterMask(const RawImage& masterMask);
	void applyMaskThreshold(float thresh);
	void growMask();

	// Subtracts a template image from the science layer.
	void subtractTemplate(const RawImage& subTemplate);
    
	// Adds an (artificial) object to the image (science) data.
	void addObject(float x, float y, float flux,
		       const PointSpreadFunc& psf);

	// Adds an object to the mask data.
	void maskObject(float x, float y, const PointSpreadFunc& psf);

	// Saves the data in each later to a file.
	void saveLayers(const std::string& path);
	void saveSci(const std::string& path);
	void saveMask(const std::string& path);
	void saveVar(const std::string& path);

	// Setter functions for the individual layers.
	void setScience(RawImage& im);
	void setMask(RawImage& im);
	void setVariance(RawImage& im);

	//pybind11::array_t<float> sciToNumpy();
	virtual void convolve(PointSpreadFunc psf) override;
	RawImage poolScience() { return science.pool(POOL_MAX); }
	RawImage poolVariance() { return variance.pool(POOL_MIN); }
	virtual ~LayeredImage() {};

private:
	void readHeader(const std::string& filePath);
	void loadLayers(const std::string& filePath);
	void readFitsImg(const char *name, float *target);
	void checkDims(RawImage& im);

	std::string fileName;
	unsigned width;
	unsigned height;
	long dimensions[2];
	unsigned pixelsPerImage;
	double captureTime;
	RawImage science;
	RawImage mask;
	RawImage variance;
};

} /* namespace kbmod */

#endif /* LAYEREDIMAGE_H_ */
