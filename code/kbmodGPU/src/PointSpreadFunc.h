/*
 * GeneratorPSF.h
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 */

#ifndef GENERATORPSF_H_
#define GENERATORPSF_H_

#include <cmath>
#include <iostream>
#include <vector>
#include "common.h"

class PointSpreadFunc
{
	public:
		PointSpreadFunc(float stdev);
		virtual ~PointSpreadFunc();
		float getStdev();
		float getSum();
		int getDim();
		int getRadius();
		int getSize();
		float* kernelData();
		void squarePSF();
		void printPSF(int debug);
	private:
		std::vector<float> kernel;
		float width;
		float sum;
		int dim;
		int radius;
};

#endif /* GENERATORPSF_H_ */
