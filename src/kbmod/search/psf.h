#ifndef PSF_H_
#define PSF_H_

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "common.h"
#include "pydocs/psf_docs.h"

namespace search {
class PSF {
public:
    PSF();  // Create a no-op PSF.
    PSF(float stdev);
    PSF(const PSF& other);  // Copy constructor
    PSF(PSF&& other);       // Move constructor
#ifdef Py_PYTHON_H
    PSF(pybind11::array_t<float> arr);
    void set_array(pybind11::array_t<float> arr);
#endif

    virtual ~PSF(){};

    // Assignment functions.
    PSF& operator=(const PSF& other);  // Copy assignment
    PSF& operator=(PSF&& other);       // Move assignment

    // Getter functions (inlined)
    float get_sum() const { return sum; }
    float get_value(int x, int y) const { return kernel[y * dim + x]; }
    int get_dim() const { return dim; }  // Length of one side of the kernel.
    int get_radius() const { return radius; }
    int get_size() const { return kernel.size(); }
    const std::vector<float>& get_kernel() const { return kernel; };
    float* data() { return kernel.data(); }

    // Computation functions.
    void square_psf();
    std::string print();

private:
    // Validates the PSF array and computes the sum.
    void calc_sum();

    std::vector<float> kernel;
    float sum;
    int dim;
    int radius;
};

} /* namespace search */

#endif /* PSF_H_ */
