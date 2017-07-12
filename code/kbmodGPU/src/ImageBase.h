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
	virtual long* getDimensions();
	virtual unsigned getPPI();
	virtual float getWidth();
	virtual float getHeight();
	virtual ~ImageBase();
};

} /* namespace kbmod */

#endif /* IMAGEBASE_H_ */
