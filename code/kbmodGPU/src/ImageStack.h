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
	virtual ~ImageStack();
	void findFiles(std::string path);
	void loadImages();
	void loadImages(std::list<std::string> files);
	void saveImages(std::string path);
	void applyMasterMask(int flags, int threshold);
	void applyMaskFlags(int flags);

private:
	void createMasterMask(int flags, int threshold);
	std::string rootPath;
	std::list<std::string> fileNames;
	std::vector<RawImage> images;
	std::vector<float> masterMask;
	std::vector<float> imageTimes;
	int width;
	int height;
	long dimensions[2];
	int pixelsPerImage;
	bool verbose;
};

#endif /* IMAGESTACK_H_ */
