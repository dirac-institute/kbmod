/*
 * ImageStack.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "ImageStack.h"

ImageStack::ImageStack(std::string path, bool verbse) {
	rootDir = dir;
	verbose = verbse;
	width = 0;
	height = 0;
	dimensions = {0, 0};
	pixelsPerImage = 0;
}

void ImageStack::loadImages()
{
	findFiles(rootPath);
	loadImages(fileNames);
}

void ImageStack::loadImages(std::list<std::string> files)
{

	getDimensions(files.front());
	// Buffers to hold the 3 image layers read by cfitsio
	float *sBuffer = new float[pixelsPerImage];
	float *vBuffer = new float[pixelsPerImage];
	float *mBuffer = new float[pixelsPerImage];

	// Load images from file
	double firstImageTime = readFitsMJD((fileNames.front()+"[0]").c_str());
	for (std::list<std::string>::iterator it=fileNames.begin();
		it != fileNames.end(); ++it)
	{
		// Read Images
		float imgTime = (readFitsMJD((*it+"[0]").c_str())-firstImageTime);
		readFitsImg((*it+"[1]").c_str(), pixelsPerImage, sBuffer);
		readFitsImg((*it+"[2]").c_str(), pixelsPerImage, vBuffer);
		readFitsImg((*it+"[3]").c_str(), pixelsPerImage, mBuffer);
		images.push_back(RawImage(sBuffer, vBuffer, mBuffer, width, height, imgTime));
	}

	delete sBuffer;
	delete vBuffer;
	delete mBuffer;

	if (verbose)
	{
		cout << "\nImage times: ";
		for (std::vector<RawImage>::iterator it=images.begin();
			it != images.end(); ++it)
		{
			cout << *it.getTime() << " ";
		}
		cout << "\n";
	}

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

		if (verbose) {
			cout << "Found " << fileNames.size()
			     << " items in " << path << "\n";
		}

}

void ImageStack::getDimensions(std::string imgPath)
{
	fitsfile *fptr1;
	int status = 0;
	int fileNotFound;
	// Read dimensions of image

	if (fits_open_file(&fptr1, (imgPath+"[1]").c_str(),
		READONLY, &status)) fits_report_error(stderr, status);
	if (fits_read_keys_lng(fptr1, "NAXIS", 1, 2, dimensions,
		&fileNotFound, &status)) fits_report_error(stderr, status);
	if (fits_close_file(fptr1, &status)) fits_report_error(stderr, status);
}

void ImageStack::applyMasterMask(int threshold)
{

}

void ImageStack::applyMaskFlags(int flag)
{

}

ImageStack::~ImageStack() {}

