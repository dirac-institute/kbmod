/*
 * RawImage.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#ifndef RAWIMAGE_H_
#define RAWIMAGE_H_

#include <vector>
#include <fitsio.h>
#include <iostream>
#include <string>
#include <assert.h>

class RawImage {
public:
	RawImage(std::string path);
	float* getSDataRef(); // Get pointer to science pixels
	float* getVDataRef(); // Get pointer to variance pixels
	float* getMDataRef(); // Get pointer to mask pixels
	void freeLayers();
	void applyMaskFlags(int flag);
	void applyMasterMask(std::vector<float> *maskPix);
	void saveSci(std::string path);
    void saveMask(std::string path);
	void saveVar(std::string path);
	static void writeFitsImg(std::string path, void *array);
	double getTime();
	float getWidth();
	float getHeight();
	virtual ~RawImage();

private:
	void readHeader();
	void loadLayers();
	void readFitsImg(const char *name, float *target);
	void mask(int flag, std::vector<float> *target, std::vector<float> *maskPix);
	bool layersLoaded;
	std::string filePath;
	std::string fileName;
	unsigned width;
	unsigned height;
	long dimensions[2];
	unsigned pixelsPerImage;
	double captureTime;
	std::vector<float> sciencePixels;
	std::vector<float> maskPixels;
	std::vector<float> variancePixels;
};

#endif /* RAWIMAGE_H_ */
