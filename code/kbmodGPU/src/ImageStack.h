/*
 * ImageStack.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <dirent.h>
#include <fitsio.h>

class ImageStack {
public:
	ImageStack(std::string path, bool verbse);
	virtual ~ImageStack();
	void findFiles(std::string path);
	void loadImages();
	void loadImages(std::list<std::string> files);
	void applyMasterMask(int threshold);
	void applyMaskFlags(int flag);

private:
	void getDimensions(std::string imgPath);
	std::string rootPath;
	std::list<std::string> fileNames;
	std::vector<RawImage> images;
	RawImage masterMask;
	int width;
	int height;
	int dimensions[2];
	int pixelsPerImage;
	bool verbose;
};

#endif /* IMAGESTACK_H_ */
