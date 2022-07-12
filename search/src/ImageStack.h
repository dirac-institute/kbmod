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

class ImageStack : public ImageBase {
public:
	ImageStack(const std::vector<std::string>& filenames);
	ImageStack(const std::vector<LayeredImage>& imgs);
	std::vector<LayeredImage>& getImages();
	unsigned imgCount() const;
	std::vector<float> getTimes();
	void setTimes(const std::vector<float>& times);
	void resetImages();
	void saveMasterMask(const std::string& path);
	void saveImages(const std::string& path);
	const RawImage& getMasterMask() const;
	std::vector<RawImage> getSciences();
	std::vector<RawImage> getMasks();
	std::vector<RawImage> getVariances();
	void applyMasterMask(int flags, int threshold);
	void applyMaskFlags(int flags, const std::vector<int>& exceptions);
	void applyMaskThreshold(float thresh);
	void growMask();
	void simpleDifference();
	virtual void convolve(PointSpreadFunc psf) override;
	unsigned getWidth() const override { return images[0].getWidth(); }
	unsigned getHeight() const override { return images[0].getHeight(); }
	long* getDimensions() override { return images[0].getDimensions(); }
	unsigned getPPI() const override { return images[0].getPPI(); }
	virtual ~ImageStack() {};

private:
	void loadImages(const std::vector<std::string>& fileNames);
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
