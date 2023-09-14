/*
 * Filtering.cpp
 *
 * Created on: Sept 2, 2022
 *
 * Helper functions for filtering results.
 */

#include "Filtering.h"
#include <math.h>

namespace search {

#ifdef HAVE_CUDA
    /* The filter_kenerls.cu functions. */
    extern "C" void sigmaGFilteredIndicesCU(float* values, int num_values, float sGL0, float sGL1,
                                            float sigmaGCoeff, float width, int* idxArray, int* minKeepIndex,
                                            int* maxKeepIndex);
#endif

/* Return the list of indices from the values array such that those elements
   pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
   sigmaGCoeff and a multiplicative factor of width. */
std::vector<int> sigmaGFilteredIndices(const std::vector<float>& values, float sGL0, float sGL1,
                                       float sigmaGCoeff, float width) {
    // Bounds check the percentile values.
    assert(sGL0 > 0.0);
    assert(sGL1 < 1.0);

    // Allocate space for the input and result.
    const int num_values = values.size();
    float values_arr[num_values];
    int idxArray[num_values];
    for (int i = 0; i < num_values; ++i) {
        values_arr[i] = values[i];
    }

    int minKeepIndex = 0;
    int maxKeepIndex = num_values - 1;

    #ifdef HAVE_CUDA
        sigmaGFilteredIndicesCU(values_arr, num_values, sGL0, sGL1, sigmaGCoeff, width, idxArray,
                                &minKeepIndex, &maxKeepIndex);
    #else
        throw std::runtime_error("Non-GPU sigmaGFilteredIndicesCU is not implemented.");
    #endif

    // Copy the result into a vector and return it.
    std::vector<int> result;
    for (int i = minKeepIndex; i <= maxKeepIndex; ++i) {
        result.push_back(idxArray[i]);
    }
    return result;
}

/* Given a set of psi and phi values,
   return a likelihood value */
double calculateLikelihoodFromPsiPhi(std::vector<double> psiValues, std::vector<double> phiValues) {
    assert(psiValues.size() == phiValues.size());
    double psiSum = 0.0;
    double phiSum = 0.0;

    for (int i = 0; i < psiValues.size(); i++) {
        psiSum += psiValues[i];
        phiSum += phiValues[i];
    }

    if (psiSum == 0.0 || phiSum <= 0.0) {
        return 0.0;
    }

    return psiSum / sqrt(phiSum);
}

} /* namespace search */
