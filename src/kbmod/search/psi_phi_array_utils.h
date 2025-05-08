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
#include "image_stack.h"
#include "layered_image.h"
#include "psi_phi_array_ds.h"

namespace search {

// Compute the min, max, and scale parameter from the a vector of image data.
std::array<float, 3> compute_scale_params_from_image_vect(const std::vector<Image>& imgs, int num_bytes);

void fill_psi_phi_array(PsiPhiArray& result_data, int num_bytes, const std::vector<Image>& psi_imgs,
                        const std::vector<Image>& phi_imgs, const std::vector<double> zeroed_times);

void fill_psi_phi_array_from_image_stack(PsiPhiArray& result_data, ImageStack& stack, int num_bytes);

} /* namespace search */

#endif /* PSI_PHI_ARRAY_UTILS_ */
