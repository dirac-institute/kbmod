/*
 * search_data_utils.h
 *
 * The utility functions for the psi/phi array. Broken out from the header
 * data structure so that it can use packages that won't be imported into the
 * CUDA kernel, such as Eigen.
 *
 * Created on: Dec 8, 2023
 */

#ifndef SEARCH_DATA_UTILS_
#define SEARCH_DATA_UTILS_

#include <cmath>
#include <stdio.h>
#include <float.h>
#include <vector>

#include "common.h"
#include "image_stack.h"
#include "layered_image.h"
#include "search_data_ds.h"
#include "raw_image.h"

namespace search {

// Compute the min, max, and scale parameter from the a vector of image data.
std::array<float, 3> compute_scale_params_from_image_vect(const std::vector<RawImage>& imgs, int num_bytes);

void fill_search_data(SearchData& result_data, int num_bytes, const std::vector<RawImage>& psi_imgs,
                      const std::vector<RawImage>& phi_imgs, const std::vector<float> zeroed_times,
                      bool debug = false);

void fill_search_data_from_image_stack(SearchData& result_data, ImageStack& stack, int num_bytes,
                                       bool debug = false);

} /* namespace search */

#endif /* SEARCH_DATA_UTILS_ */
