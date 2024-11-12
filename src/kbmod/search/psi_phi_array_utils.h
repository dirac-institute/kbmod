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
#include "raw_image.h"

namespace search {

void fill_psi_phi_array(PsiPhiArray& result_data, const std::vector<RawImage>& psi_imgs,
                        const std::vector<RawImage>& phi_imgs, const std::vector<double> zeroed_times);

void fill_psi_phi_array_from_image_stack(PsiPhiArray& result_data, ImageStack& stack);

} /* namespace search */

#endif /* PSI_PHI_ARRAY_UTILS_ */
