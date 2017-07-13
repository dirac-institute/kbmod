/*
 * ImageBase.h
 *
 *  Created on: Jul 12, 2017
 *      Author: kbmod-usr
 */

#ifndef IMAGEBASE_H_
#define IMAGEBASE_H_

#include "PointSpreadFunc.h"

namespace kbmod {

class ImageBase {
public:
	ImageBase();
	virtual void convolve(PointSpreadFunc psf);
	virtual unsigned getWidth() = 0;
	virtual unsigned getHeight() = 0;
	virtual long* getDimensions() = 0;
	virtual unsigned getPPI() = 0;
	virtual ~ImageBase();
};

} /* namespace kbmod */

#endif /* IMAGEBASE_H_ */
