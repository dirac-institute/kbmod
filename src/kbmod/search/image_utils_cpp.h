#ifndef IMAGE_UTILS_CPP_H_
#define IMAGE_UTILS_CPP_H_

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

#include "common.h"

namespace search {

using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageRef = Eigen::Ref<Image>;
using ImageIRef = Eigen::Ref<Image>;

// Functions for convolution.
Image convolve_image_cpu(Image& img, Image& psf);
Image convolve_image_gpu(Image& img, Image& psf);
// allow_gpu=false forces the CPU kernel even when a GPU is present. The GPU path costs a
// full round trip (alloc, H2D, launch, D2H, free) per call, which dominates for small
// images: measured ~11 ms vs ~1.2 us at 5x17. Defaults preserve existing behaviour.
Image convolve_image(Image& img, Image& psf, bool allow_gpu = true);
Image square_psf_values(Image& given_psf);

// Functions for psi and phi generation.
Image generate_psi(Image& sci, Image& var, Image& psf, bool allow_gpu = true);
Image generate_phi(Image& var, Image& psf, bool allow_gpu = true);

} /* namespace search */

#endif /* IMAGE_UTILS_CPP_H_ */
