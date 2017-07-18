/*
 * ImageStack.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <vector>
#include <dirent.h>
#include <string>
#include <list>
#include <iostream>
#include "LayeredImage.h"

namespace kbmod {

class ImageStack : public ImageBase {
public:
	ImageStack(std::list<std::string> files);
	std::vector<LayeredImage> getImages();
	int imgCount();
	std::vector<float> getTimes();
	void resetImages();
	void saveSci(std::string path);
	void saveMask(std::string path);
	void saveVar(std::string path);
	void applyMasterMask(int flags, int threshold);
	void applyMaskFlags(int flags);
	virtual void convolve(PointSpreadFunc psf) override;
	unsigned getWidth() override { return images[0].getWidth(); }
	unsigned getHeight() override { return images[0].getHeight(); }
	long* getDimensions() override { return images[0].getDimensions(); }
	unsigned getPPI() override { return images[0].getPPI(); }
	virtual ~ImageStack() {};

private:
	void loadImages();
	void createMasterMask(int flags, int threshold);
	std::list<std::string> fileNames;
	std::vector<LayeredImage> images;
	RawImage masterMask;
	std::vector<float> imageTimes;
	bool verbose;
};

} /* namespace kbmod */

#endif /* IMAGESTACK_H_ */
