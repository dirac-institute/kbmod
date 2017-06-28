/*
 * ImageStack.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "ImageStack.h"

ImageStack::ImageStack(std::string path, bool verbse) {
	rootPath = path;
	verbose = verbse;
	fileNames = std::list<std::string>();
	width = 0;
	height = 0;
	//dimensions = {0, 0};
	pixelsPerImage = 0;
}

void ImageStack::loadImages()
{
	findFiles(rootPath);
	loadImages(fileNames);
}

void ImageStack::loadImages(std::list<std::string> files)
{
	// Load images from file
	for (auto& i : files)
	{
		images.push_back(RawImage(i));
		if (verbose) std::cout << "." << std::flush;
	}
	if (verbose) std::cout << "\n";

	// Should do a test here to make sure all images are same dimensions
	width = images[0].getWidth();
	height = images[0].getHeight();
	pixelsPerImage = width*height;
	dimensions[0] = width;
	dimensions[1] = height;

	// Load image times
	double initialTime = images[0].getTime();
	imageTimes = std::vector<float>();
	for (auto& i : images)
	{
		imageTimes.push_back(float(i.getTime()-initialTime));
	}

	if (verbose)
	{
		std::cout << "\nImage times: ";
		for (auto& i : imageTimes)
		{
			std::cout << i << " ";
		}
		std::cout << "\n";
	}
}

void ImageStack::saveSci(std::string path)
{
	for (auto& i : images) i.saveSci(path);
}
void ImageStack::saveMask(std::string path)
{
	for (auto& i : images) i.saveMask(path);
}
void ImageStack::saveVar(std::string path)
{
	for (auto& i : images) i.saveVar(path);
}
void ImageStack::savePsi(std::string path)
{
	for (auto& i : images) i.savePsi(path);
}
void ImageStack::savePhi(std::string path)
{
	for (auto& i : images) i.savePhi(path);
}

/* Read list of files from directory and get their dimensions  */
void ImageStack::findFiles(std::string path)
{
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir (path.c_str())) != NULL) {
			/* add all the files and directories within directory */
			while ((ent = readdir (dir)) != NULL) {
				std::string current = ent->d_name;
				if (current != "." && current != "..")
				{
					fileNames.push_back(path+current);
				}
			}
		closedir (dir);
		}

		// Filter out files without ".fits" in the name
		fileNames.remove_if( [] (std::string s)
		{
			return s.find(".fits") == std::string::npos;
		});

		fileNames.sort();

		if (fileNames.size() < 1) {
			std::cout << "No fits images found!\n";
		}
		if (verbose) {
			std::cout << "Found " << fileNames.size()
			     << " items in " << path << "\n";
		}
}

void ImageStack::applyMaskFlags(int flags)
{
	for (auto& i : images)
	{
		i.applyMaskFlags(flags);
	}
}

void ImageStack::applyMasterMask(int flags, int threshold)
{
	createMasterMask(flags, threshold);
	for (auto& i : images)
	{
		i.applyMasterMask(&masterMask);
	}
}

void ImageStack::createMasterMask(int flags, int threshold)
{
	// Initialize masterMask to 0.0s
	masterMask = std::vector<float>(pixelsPerImage, 0.0);
	for (unsigned int img=0; img<images.size(); ++img)
	{
		float *imgMask = images[img].getMDataRef();
		for (unsigned int pixel=0; pixel<pixelsPerImage; ++pixel)
		{
			if (flags & static_cast<int>(imgMask[pixel]) != 0)
				masterMask[pixel]++;
		}
	}

	// Set all pixels below threshold to 0 and all above to 1
	float fThreshold = static_cast<float>(threshold);
	for (auto& p : masterMask)
	{
		p = p < fThreshold ? 0.0 : 1.0;
	}
}

ImageStack::~ImageStack() {}

