/*
 * ImageStack.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "ImageStack.h"

namespace kbmod {

ImageStack::ImageStack(std::list<std::string> files, bool verbse) {
	verbose = verbse;
	fileNames = files;
	loadImages();
}

void ImageStack::loadImages()
{

	if (fileNames.size()==0)
	{
		std::cout << "No files provided" << "\n";
	}

	// Load images from file
	for (auto& i : fileNames)
	{
		images.push_back(LayeredImage(i));
		if (verbose) std::cout << "." << std::flush;
	}
	if (verbose) std::cout << "\n";

	// Should do a test here to make sure all images are same dimensions
	/*
	width = images[0].getWidth();
	height = images[0].getHeight();
	pixelsPerImage = width*height;
	dimensions[0] = width;
	dimensions[1] = height;
	*/

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

std::vector<LayeredImage> ImageStack::getImages()
{
	return images;
}

int ImageStack::imgCount()
{
	return images.size();
}

std::vector<float> ImageStack::getTimes()
{
	return imageTimes;
}

void ImageStack::freeImages()
{
	images = std::vector<LayeredImage>();
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


/* Read list of files from directory and get their dimensions  * /
void ImageStack::findFiles(std::string path)
{
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir (path.c_str())) != NULL) {
			/* add all the files and directories within directory * /
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
		std::cout << "Files before filtering: " << fileNames.size() << "\n";
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
*/

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
		i.applyMasterMask(masterMask);
	}
}

void ImageStack::createMasterMask(int flags, int threshold)
{
	// Initialize masterMask to 0.0s
	std::vector<float> masterM(getPPI());
	for (unsigned int img=0; img<images.size(); ++img)
	{
		float *imgMask = images[img].getMDataRef();
		for (unsigned int pixel=0; pixel<getPPI(); ++pixel)
		{
			if ((flags & static_cast<int>(imgMask[pixel])) != 0)
				masterM[pixel]++;
		}
	}

	// Set all pixels below threshold to 0 and all above to 1
	float fThreshold = static_cast<float>(threshold);
	for (unsigned int p=0; p<getPPI(); ++p)
	{
		masterM[p] = masterM[p] < fThreshold ? 0.0 : 1.0;
	}

	masterMask(getWidth(), getHeight(), masterM.data());

}

ImageStack::~ImageStack() {}

} /* namespace kbmod */

