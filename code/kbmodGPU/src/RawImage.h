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
	float* getPsiDataRef(); //   pointer to psi pixels
	float* getPhiDataRef(); //   pointer to phi pixels
	void freeLayers();
	void applyMaskFlags(int flag);
	void applyMasterMask(std::vector<float> *maskPix);
	void saveSci(std::string path);
    void saveMask(std::string path);
	void saveVar(std::string path);
	double getTime();
	float getWidth();
	float getHeight();
	virtual ~RawImage();

private:
	void readHeader();
	void loadLayers();
	void readFitsImg(const char *name, float *target);
	void writeFitsImg(std::string path, void *array);
	void mask(int flag, std::vector<float> *target, std::vector<float> *maskPix);
	bool layersLoaded;
	std::string filePath;
	std::string fileName;
	int width;
	int height;
	long dimensions[2];
	int pixelsPerImage;
	double captureTime;
	std::vector<float> sciencePixels;
	std::vector<float> maskPixels;
	std::vector<float> variancePixels;
};

#endif /* RAWIMAGE_H_ */
