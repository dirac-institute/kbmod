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
#include "common.h"


namespace image_base {
  class PointSpreadFunc {
  public:
    PointSpreadFunc(float stdev);
    PointSpreadFunc(const PointSpreadFunc& other);  // Copy constructor
    PointSpreadFunc(PointSpreadFunc&& other);       // Move constructor
#ifdef Py_PYTHON_H
    PointSpreadFunc(pybind11::array_t<float> arr);
    void set_array(pybind11::array_t<float> arr);
#endif
    virtual ~PointSpreadFunc(){};

    // Assignment functions.
    PointSpreadFunc& operator=(const PointSpreadFunc& other);  // Copy assignment
    PointSpreadFunc& operator=(PointSpreadFunc&& other);       // Move assignment

    // Getter functions (inlined)
    float get_stdev() const { return width; }
    float get_sum() const { return sum; }
    float get_value(int x, int y) const { return kernel[y * dim + x]; }
    int get_dim() const { return dim; }
    int get_radius() const { return radius; }
    int get_size() const { return kernel.size(); }
    const std::vector<float>& get_kernel() const { return kernel; };
    float* data() { return kernel.data(); }

    // Computation functions.
    void calc_sum();
    void square_psf();
    std::string print();

  private:
    std::vector<float> kernel;
    float width;
    float sum;
    int dim;
    int radius;
  };
} /* namespace image_base */

#endif /* POINTSPREADFUNC_H_ */
