#include "filtering.h"
#include <math.h>

namespace search {

#ifdef HAVE_CUDA
  /* The filter_kenerls.cu functions. */
  extern "C" void SigmaGFilteredIndicesCU(float *values, int num_values, float sgl0, float sgl1, float sg_coeff,
					  float width, int *idx_array, int *min_keep_idx, int *max_keep_idx);
#endif

  /* Return the list of indices from the values array such that those elements
     pass the sigmaG filtering defined by percentiles [sgl0, sgl1] with coefficient
     sigma_g_coeff and a multiplicative factor of width. */
  std::vector<int> sigmaGFilteredIndices(const std::vector<float> &values, float sgl0, float sgl1,
					 float sigma_g_coeff, float width) {
    // Bounds check the percentile values.
    assert(sgl0 > 0.0);
    assert(sgl1 < 1.0);

    // Allocate space for the input and result.
    const int num_values = values.size();
    float values_arr[num_values];
    int idx_array[num_values];
    for (int i = 0; i < num_values; ++i) {
      values_arr[i] = values[i];
    }

    int min_keep_idx = 0;
    int max_keep_idx = num_values - 1;

#ifdef HAVE_CUDA
    SigmaGFilteredIndicesCU(values_arr, num_values, sgl0, sgl1, sigma_g_coeff, width, idx_array,
                            &min_keep_idx, &max_keep_idx);
#else
    throw std::runtime_error("Non-GPU SigmaGFilteredIndicesCU is not implemented.");
#endif

    // Copy the result into a vector and return it.
    std::vector<int> result;
    for (int i = min_keep_idx; i <= max_keep_idx; ++i) {
      result.push_back(idx_array[i]);
    }
    return result;
  }

  /* Given a set of psi and phi values,
     return a likelihood value */
  double calculateLikelihoodFromPsiPhi(std::vector<double> psi_values, std::vector<double> phi_values) {
    assert(psi_values.size() == phi_values.size());
    double psi_sum = 0.0;
    double phi_sum = 0.0;

    for (int i = 0; i < psi_values.size(); i++) {
      psi_sum += psi_values[i];
      phi_sum += phi_values[i];
    }

    if (psi_sum == 0.0 || phi_sum <= 0.0) {
      return 0.0;
    }

    return psi_sum / sqrt(phi_sum);
  }
    
} /* namespace search */
