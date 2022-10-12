/*
 * GeneratorPSF.cpp
 *
 *  Created on: Nov 12, 2016
 *      Author: peter
 */

#include "PointSpreadFunc.h"

namespace kbmod {

PointSpreadFunc::PointSpreadFunc(float stdev) {
    width = stdev;
    float simpleGauss[MAX_KERNEL_RADIUS];
    double psfCoverage = 0.0;
    double normFactor = stdev * sqrt(2.0);
    int i = 0;

    // Create 1D gaussian array
    while (psfCoverage < 0.98 && i < MAX_KERNEL_RADIUS) {
        float currentBin =
                0.5 * (std::erf((float(i) + 0.5) / normFactor) - std::erf((float(i) - 0.5) / normFactor));
        simpleGauss[i] = currentBin;
        if (i == 0) {
            psfCoverage += currentBin;
        } else {
            psfCoverage += 2.0 * currentBin;
        }
        i++;
    }

    radius = i - 1;  // This value is good for
    dim = 2 * radius + 1;

    // Create 2D gaussain by multiplying with itself
    kernel = std::vector<float>();
    for (int ii = 0; ii < dim; ++ii) {
        for (int jj = 0; jj < dim; ++jj) {
            float current = simpleGauss[abs(radius - ii)] * simpleGauss[abs(radius - jj)];
            kernel.push_back(current);
        }
    }
    calcSum();
}

PointSpreadFunc::PointSpreadFunc(const PointSpreadFunc& other) {
    kernel = other.getKernel();
    dim = other.getDim();
    radius = other.getRadius();
    width = other.getStdev();
    calcSum();
}

#ifdef Py_PYTHON_H
PointSpreadFunc::PointSpreadFunc(pybind11::array_t<float> arr) { setArray(arr); }

void PointSpreadFunc::setArray(pybind11::array_t<float> arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2)
        throw std::runtime_error(
                "Array must have 2 dimensions. (It "
                " must also be a square with odd dimensions)");

    if (info.shape[0] != info.shape[1])
        throw std::runtime_error(
                "Array must be square (x-dimension == y-dimension)."
                "It also must have odd dimensions.");
    float* pix = static_cast<float*>(info.ptr);
    dim = info.shape[0];
    if (dim % 2 == 0)
        throw std::runtime_error(
                "Array dimension must be odd. The "
                "middle of an even numbered array is ambiguous.");
    radius = dim / 2;  // Rounds down
    sum = 0.0;
    kernel = std::vector<float>(pix, pix + dim * dim);
    calcSum();
    width = 0.0;
}
#endif

void PointSpreadFunc::calcSum() {
    sum = 0.0;
    for (auto& i : kernel) sum += i;
}

void PointSpreadFunc::squarePSF() {
    for (float& i : kernel) {
        i = i * i;
    }
    calcSum();
}

std::string PointSpreadFunc::printPSF() {
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(3);
    for (int row = 0; row < dim; ++row) {
        ss << "| ";
        for (int col = 0; col < dim; ++col) {
            ss << kernel[row * dim + col] << " | ";
        }
        ss << "\n ";
        for (int space = 0; space < dim * 8 - 1; ++space) ss << "-";
        ss << "\n";
    }
    ss << 100.0 * sum << "% of PSF contained within kernel\n";
    return ss.str();
}

} /* namespace kbmod */
