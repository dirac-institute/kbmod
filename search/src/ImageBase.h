/*
 * ImageBase.h
 *
 *  Created on: Jul 12, 2017
 *      Author: kbmod-usr
 *
 * A base class for different types of images used by kbmod.
 */

#ifndef IMAGEBASE_H_
#define IMAGEBASE_H_

#include "PointSpreadFunc.h"

namespace kbmod {

class ImageBase {
public:
	ImageBase() {};
	virtual void convolve(PointSpreadFunc psf) = 0;
	virtual unsigned getWidth() const = 0;
	virtual unsigned getHeight() const = 0;
	virtual long* getDimensions() = 0;
	virtual unsigned getPPI() const = 0;
	virtual ~ImageBase() {};
};

} /* namespace kbmod */

#endif /* IMAGEBASE_H_ */
