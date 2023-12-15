/* Helper functions for testing functions in the .cu files from Python. */

#include <vector>

namespace search {
#ifdef HAVE_CUDA
/* The filter_kenerls.cu functions. */
extern "C" void SigmaGFilteredIndicesCU(float *values, int num_values, float sgl0, float sgl1, float sg_coeff,
                                        float width, int *idx_array, int *min_keep_idx, int *max_keep_idx);
#endif

/* Used for testing SigmaGFilteredIndicesCU for python
 *
 * Return the list of indices from the values array such that those elements
 * pass the sigmaG filtering defined by percentiles [sgl0, sgl1] with coefficient
 * sigma_g_coeff and a multiplicative factor of width.
 *
 * The vector values is passed by value to create a local copy which will be modified by
 * SigmaGFilteredIndicesCU.
 */
std::vector<int> sigmaGFilteredIndices(std::vector<float> values, float sgl0, float sgl1, float sigma_g_coeff,
                                       float width) {
    int num_values = values.size();
    std::vector<int> idx_array(num_values, 0);
    int min_keep_idx = 0;
    int max_keep_idx = num_values - 1;

#ifdef HAVE_CUDA
    SigmaGFilteredIndicesCU(values.data(), num_values, sgl0, sgl1, sigma_g_coeff, width, idx_array.data(),
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

} /* namespace search */
