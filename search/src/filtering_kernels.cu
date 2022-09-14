/*
 * filtering_kernels.cu
 *
 *  Created on: September 14th, 2022
 *
 */

#ifndef FILTERING_KERNELS_CU_
#define FILTERING_KERNELS_CU_

#include "common.h"
#include <helper_cuda.h>
#include <stdio.h>
#include <float.h>

namespace kbmod {

extern "C" __device__ __host__ 
void sigmaGFilteredIndicesCU(float* values, int num_values,
                             float sGL0, float sGL1, float sigmaGCoeff,
                             int* idxArray, int* minKeepIndex, int* maxKeepIndex)
{
    // Initialize the index array.
    for (int j = 0; j < num_values; j++)
    {
        idxArray[j] = j;
    }

    // Sort the the indexes (idxArray) of values in ascending order.
    int tmpSortIdx;
    for (int j = 0; j < num_values; j++)
    {
        for (int k = j+1; k < num_values; k++)
        {
            if (values[idxArray[j]] > values[idxArray[k]])
            {
                tmpSortIdx = idxArray[j];
                idxArray[j] = idxArray[k];
                idxArray[k] = tmpSortIdx;
            }
        }
    }

    // Compute the index of each of the percent values in values
    // from the given bounds sGL0, 0.5 (median), and sGL1.
    const int pct_L = int((num_values + 1) * sGL0 + 0.5) - 1;
    const int pct_H = int((num_values + 1) * sGL1 + 0.5) - 1;
    const int median_ind = int((num_values + 1) * 0.5 + 0.5) - 1;

    // Compute the values that are +/- 2*sigmaG from the median.
    float sigmaG = sigmaGCoeff * (values[idxArray[pct_H]]
                                  - values[idxArray[pct_L]]);
    float minValue = values[idxArray[median_ind]] - 2 * sigmaG;
    float maxValue = values[idxArray[median_ind]] + 2 * sigmaG;

    // Find the index of the first value >= minValue.
    int start = 0;
    while ((start < median_ind) && (values[idxArray[start]] < minValue))
    {
        ++start;
    }
    *minKeepIndex = start;

    // Find the index of the last value <= maxValue.
    int end = median_ind + 1;
    while ((end < num_values) && (values[idxArray[end]] <= maxValue))
    {
        ++end;
    }
    *maxKeepIndex = end - 1;
}

} /* namespace kbmod */

#endif /* FILTERING_KERNELS_CU_ */
