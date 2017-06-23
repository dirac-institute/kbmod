/*
 * RawImage.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#ifndef RAWIMAGE_H_
#define RAWIMAGE_H_

#include <vector>

class RawImage {
public:
	RawImage(float *sData, float *vData,
			 float *mData, int x, int y, float time);
	virtual ~RawImage();
	float* getSDataRef(); // Get pointer to science pixels
	float* getVDataRef(); // Get pointer to variance pixels
	float* getMDataRef(); // Get pointer to mask pixels
	float getTime();

private:
	int width;
	int height;
	int pixelsPerImg;
	float captureTime;
	std::vector<float> sciencePixels;
	std::vector<float> variancePixels;
	std::vector<float> maskPixels;
};

#endif /* RAWIMAGE_H_ */
