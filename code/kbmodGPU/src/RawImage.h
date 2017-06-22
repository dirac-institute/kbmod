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
	RawImage(float *data, int x, int y);
	virtual ~RawImage();
	float* getPixelsRef();

private:
	int width;
	int height;
	std::vector<float> pixels;
};

#endif /* RAWIMAGE_H_ */
