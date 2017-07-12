/*
 * LayeredImage.h
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 */

#ifndef LAYEREDIMAGE_H_
#define LAYEREDIMAGE_H_

#include <vector>
#include <fitsio.h>
#include <iostream>
#include <string>
#include <assert.h>
#include "common.h"

namespace kbmod {

class LayeredImage : public ImageBase {
public:
	LayeredImage(std::string path);
	float* getSDataRef(); // Get pointer to science pixels
	float* getVDataRef(); // Get pointer to variance pixels
	float* getMDataRef(); // Get pointer to mask pixels
	void applyMaskFlags(int flag);
	void applyMasterMask(RawImage masterMask);
	void saveSci(std::string path);
    void saveMask(std::string path);
	void saveVar(std::string path);
	double getTime();
private:
	void readHeader();
	void loadLayers();
	void readFitsImg(const char *name, float *target);
	std::string filePath;
	std::string fileName;
	double captureTime;
	RawImage science;
	RawImage mask;
	RawImage variance;

	virtual ~LayeredImage();
};

} /* namespace kbmod */

#endif /* LAYEREDIMAGE_H_ */
