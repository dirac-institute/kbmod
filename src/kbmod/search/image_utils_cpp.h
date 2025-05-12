#ifndef IMAGE_UTILS_CPP_H_
#define IMAGE_UTILS_CPP_H_

#include <vector>
#include <float.h>
#include <iostream>
#include <stdexcept>
#include <math.h>

#include <Eigen/Core>

#include "common.h"

namespace search {

using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageRef = Eigen::Ref<Image>;
using ImageIRef = Eigen::Ref<Image>;

// Functions for convolution.
Image convolve_image_cpu(Image& img, Image& psf);
Image convolve_image_gpu(Image& img, Image& psf);
Image convolve_image(Image& img, Image& psf);
Image square_psf(Image& given_psf);

// Functions for psi and phi generation.
Image generate_psi(Image& sci, Image& var, Image& psf);
Image generate_phi(Image& var, Image& psf);

} /* namespace search */

#endif /* IMAGE_UTILS_CPP_H_ */
