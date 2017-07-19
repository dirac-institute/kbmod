/*
 * GeneratorPSF.h
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 */

#ifndef POINTSPREADFUNC_H_
#define POINTSPREADFUNC_H_

#include <cmath>
#include <iostream>
#include <vector>
#include "common.h"

namespace kbmod {

class PointSpreadFunc
{
	public:
		PointSpreadFunc(float stdev);
		virtual ~PointSpreadFunc() {};
		float getStdev() { return width; }
		float getSum() { return sum; }
		int getDim() { return dim; }
		int getRadius() { return radius; }
		int getSize() { return kernel.size(); }
		std::vector<float> getKernel() { return kernel; };
		float* kernelData() { return kernel.data(); }
		void squarePSF();
		void printPSF();
	private:
		std::vector<float> kernel;
		float width;
		float sum;
		int dim;
		int radius;
};

} /* namespace kbmod */

#endif /* POINTSPREADFUNC_H_ */
