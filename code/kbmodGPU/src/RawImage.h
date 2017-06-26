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

class RawImage {
public:
	RawImage(std::string path);
	virtual ~RawImage();
	float* getSDataRef(); // Get pointer to science pixels
	float* getVDataRef(); // Get pointer to variance pixels
	float* getMDataRef(); // Get pointer to mask pixels
	void applyMaskFlags(int flag);
	float getTime();
	float getWidth();
	float getHeight();

private:
	void readHeader();
	void readFitsImg(const char *name, float *target);
	std::string filePath;
	int width;
	int height;
	long dimensions[2];
	int pixelsPerImage;
	float captureTime;
	std::vector<float> sciencePixels;
	std::vector<float> variancePixels;
	std::vector<float> maskPixels;
};

#endif /* RAWIMAGE_H_ */
