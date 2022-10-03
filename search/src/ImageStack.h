/*
 * ImageStack.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 * ImageStack stores a series of LayeredImages from different times. 
 */

#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <vector>
#include <dirent.h>
#include <string>
#include <list>
#include <iostream>
#include <stdexcept>
#include "LayeredImage.h"

namespace kbmod {

class ImageStack {
public:
	ImageStack(const std::vector<std::string>& filenames,
	           const std::vector<PointSpreadFunc>& psfs);
	ImageStack(const std::vector<LayeredImage>& imgs);

	// Simple getters.
	unsigned imgCount() const;
	unsigned getWidth() const { return images[0].getWidth(); }
	unsigned getHeight() const { return images[0].getHeight(); }
	long* getDimensions() { return images[0].getDimensions(); }
	unsigned getPPI() const { return images[0].getPPI(); }
	const std::vector<float>& getTimes() const;
	float * getTimesDataRef();
	LayeredImage& getSingleImage(int index);

	// Simple setters.
	void setTimes(const std::vector<float>& times);
	void resetImages();

	// Get a vector of images or layers.
	std::vector<LayeredImage>& getImages();
	std::vector<RawImage> getSciences();
	std::vector<RawImage> getMasks();
	std::vector<RawImage> getVariances();

	// Apply makes to all the images.
	void applyMasterMask(int flags, int threshold);
	void applyMaskFlags(int flags, const std::vector<int>& exceptions);
	void applyMaskThreshold(float thresh);
	void growMask(int steps);
	const RawImage& getMasterMask() const;

	void convolvePSF();
	void simpleDifference();

	// Save data to files.
	void saveMasterMask(const std::string& path);
	void saveImages(const std::string& path);

	virtual ~ImageStack() {};

private:
	void loadImages(const std::vector<std::string>& fileNames,
	                const std::vector<PointSpreadFunc>& psfs);
	void extractImageTimes();
	void setTimeOrigin();
	void createMasterMask(int flags, int threshold);
	void createTemplate();
	std::vector<LayeredImage> images;
	RawImage masterMask;
	RawImage avgTemplate;
	std::vector<float> imageTimes;
	bool verbose;
};

} /* namespace kbmod */

#endif /* IMAGESTACK_H_ */
