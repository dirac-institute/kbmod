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

class ImageStack {
public:
	ImageStack(std::string path, bool verbse);
	void findFiles(std::string path);
	void loadImages();
	void loadImages(std::list<std::string> files);
	std::vector<RawImage> getImages();
	int imgCount();
	unsigned getPPI();
	long* getDimensions();
	std::vector<float> getTimes();
	void freeImages();
	void saveSci(std::string path);
	void saveMask(std::string path);
	void saveVar(std::string path);
	void applyMasterMask(int flags, int threshold);
	void applyMaskFlags(int flags);
	virtual ~ImageStack();

private:
	void createMasterMask(int flags, int threshold);
	std::string rootPath;
	std::list<std::string> fileNames;
	std::vector<RawImage> images;
	std::vector<float> masterMask;
	std::vector<float> imageTimes;
	unsigned width;
	unsigned height;
	long dimensions[2];
	unsigned pixelsPerImage;
	bool verbose;
};

#endif /* IMAGESTACK_H_ */
