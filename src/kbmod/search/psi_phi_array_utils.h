/*
 * psi_phi_array_utils.h
 *
 * The utility functions for the psi/phi array. Broken out from the header
 * data structure so that it can use packages that won't be imported into the
 * CUDA kernel, such as Eigen.
 * 
 * Created on: Dec 8, 2023
 */

#ifndef PSI_PHI_ARRAY_UTILS_
#define PSI_PHI_ARRAY_UTILS_

#include <cmath>
#include <stdio.h>
#include <float.h>
#include <vector>

#include "common.h"
#include "psi_phi_array_ds.h"
#include "raw_image.h"

namespace search {

// Compute the min, max, and scale parameter from the a vector of image data.
std::array<float, 3>  compute_scale_params_from_image_vect(const std::vector<RawImage>& imgs, int num_bytes);

void fill_psi_phi_array(PsiPhiArray& result_data,
                        const std::vector<RawImage>& psi_imgs, 
                        const std::vector<RawImage>& phi_imgs);

} /* namespace search */

#endif /* PSI_PHI_ARRAY_UTILS_ */
