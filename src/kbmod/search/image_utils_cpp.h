#ifndef IMAGE_UTILS_CPP_H_
#define IMAGE_UTILS_CPP_H_

#include <vector>
#include <float.h>
#include <iostream>
#include <stdexcept>
#include <math.h>

#include <Eigen/Core>

namespace search {

using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageRef = Eigen::Ref<Image>;
using ImageIRef = Eigen::Ref<Image>;

// Functions for convolution.
Image convolve_image_cpu(Image& img, Image& psf);
Image convolve_image_gpu(Image& img, Image& psf);
Image convolve_image(Image& img, Image& psf);

} /* namespace search */

#endif /* IMAGE_UTILS_CPP_H_ */
