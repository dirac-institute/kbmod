/*
 * KernelsWrapper.cpp
 *
 * Created on: Sept 2, 2022
 *
 * An interface to access some helper functions in the
 * kernels.cu file for better testing.
 */

#include "KernelsWrapper.h"

namespace kbmod {

/* The kernel.cu functions to test. The shouldn't need to be accessed outside 
   other functions in kernels.cu and the testing functions in these files. 
*/
extern "C" void sigmaGFilteredIndicesCU(float* values, int num_values,
        float sGL0, float sGL1, float sigmaGCoeff,
        int* idxArray, int* minKeepIndex, int* maxKeepIndex);

/* Return the list of indices from the values array such that those elements
   pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
   sigmaGCoeff */
std::vector<int> deviceSigmaGFilteredIndices(const std::vector<float>& values,
                                             float sGL0, float sGL1, float sigmaGCoeff)
{
    const int num_values = values.size();
    float values_arr[num_values];
    int idxArray[num_values];

    for (int i = 0; i < num_values; ++i) {
        values_arr[i] = values[i];
    }

    int minKeepIndex = 0;
    int maxKeepIndex = num_values - 1;
    sigmaGFilteredIndicesCU(values_arr, num_values, sGL0, sGL1, sigmaGCoeff,
                            idxArray, &minKeepIndex, &maxKeepIndex);

    std::vector<int> result;
    for (int i = minKeepIndex; i <= maxKeepIndex; ++i) {
        result.push_back(idxArray[i]);
    }

    return result;
}
    
} /* namespace kbmod */
