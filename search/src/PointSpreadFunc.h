/*
 * PointSpreadFunc.h
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 *
 * A class for working with point spread functions.
 */

#ifndef POINTSPREADFUNC_H_
#define POINTSPREADFUNC_H_

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#ifdef Py_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif
#include "common.h"

namespace kbmod {

class PointSpreadFunc
{
	public:
		PointSpreadFunc(float stdev);
		PointSpreadFunc(PointSpreadFunc& other);
#ifdef Py_PYTHON_H
		PointSpreadFunc(pybind11::array_t<float> arr);
		void setArray(pybind11::array_t<float> arr);
#endif
		virtual ~PointSpreadFunc() {};
		float getStdev() { return width; }
		void calcSum();
		float getSum() { return sum; }
		int getDim() { return dim; }
		int getRadius() { return radius; }
		int getSize() { return kernel.size(); }
		std::vector<float> getKernel() { return kernel; };
		float* kernelData() { return kernel.data(); }
		void squarePSF();
		std::string printPSF();
		// void normalize(); ???
	private:
		std::vector<float> kernel;
		float width;
		float sum;
		int dim;
		int radius;
};

} /* namespace kbmod */

#endif /* POINTSPREADFUNC_H_ */
