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

namespace search {

class PointSpreadFunc {
public:
    PointSpreadFunc();  // Create a no-op PSF.
    PointSpreadFunc(float stdev);
    PointSpreadFunc(const PointSpreadFunc& other);  // Copy constructor
    PointSpreadFunc(PointSpreadFunc&& other);       // Move constructor
#ifdef Py_PYTHON_H
    PointSpreadFunc(pybind11::array_t<float> arr);
    void setArray(pybind11::array_t<float> arr);
#endif
    virtual ~PointSpreadFunc(){};

    // Assignment functions.
    PointSpreadFunc& operator=(const PointSpreadFunc& other);  // Copy assignment
    PointSpreadFunc& operator=(PointSpreadFunc&& other);       // Move assignment

    // Getter functions (inlined)
    float getStdev() const { return width; }
    float getSum() const { return sum; }
    float getValue(int x, int y) const { return kernel[y * dim + x]; }
    int getDim() const { return dim; }
    int getRadius() const { return radius; }
    int getSize() const { return kernel.size(); }
    const std::vector<float>& getKernel() const { return kernel; };
    float* kernelData() { return kernel.data(); }

    // Computation functions.
    void calcSum();
    void squarePSF();
    std::string printPSF();

private:
    std::vector<float> kernel;
    float width;
    float sum;
    int dim;
    int radius;
};

} /* namespace search */

#endif /* POINTSPREADFUNC_H_ */
