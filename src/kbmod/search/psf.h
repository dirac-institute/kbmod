/*
 * PSF.h
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

class PSF {
public:
    PSF();  // Create a no-op PSF.
    PSF(float stdev);
    PSF(const PSF& other);  // Copy constructor
    PSF(PSF&& other);       // Move constructor
#ifdef Py_PYTHON_H
    PSF(pybind11::array_t<float> arr);
    void setArray(pybind11::array_t<float> arr);
#endif
    virtual ~PSF(){};

    // Assignment functions.
    PSF& operator=(const PSF& other);  // Copy assignment
    PSF& operator=(PSF&& other);       // Move assignment

    // Getter functions (inlined)
    float get_std() const { return width; }
    float get_sum() const { return sum; }
    float getValue(int x, int y) const { return kernel[y * dim + x]; }
    int get_dim() const { return dim; }
    int get_radius() const { return radius; }
    int get_size() const { return kernel.size(); }
    const std::vector<float>& get_kernel() const { return kernel; };
    float* kernelData() { return kernel.data(); }

    // Computation functions.
    void calcSum();
    void squarePSF();
    std::string print();

private:
    std::vector<float> kernel;
    float width;
    float sum;
    int dim;
    int radius;
};

} /* namespace search */

#endif /* POINTSPREADFUNC_H_ */
