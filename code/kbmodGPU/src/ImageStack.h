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
#include "RawImage.h"

namespace kbmod {

class ImageStack : public ImageBase {
public:
	ImageStack(std::list<std::string> files, bool verbse);
	std::vector<LayeredImage> getImages();
	int imgCount();
	std::vector<float> getTimes();
	void freeImages();
	void saveSci(std::string path);
	void saveMask(std::string path);
	void saveVar(std::string path);
	void applyMasterMask(int flags, int threshold);
	void applyMaskFlags(int flags);
	virtual ~ImageStack();

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
