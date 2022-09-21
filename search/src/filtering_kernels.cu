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
#include <cmath>

namespace kbmod {

extern "C" __device__ __host__ 
void sigmaGFilteredIndicesCU(float* values, int num_values,
                             float sGL0, float sGL1, float sigmaGCoeff,
                             float width, int* idxArray,
                             int* minKeepIndex, int* maxKeepIndex)
{
    // Clip the percentiles to [0.01, 99.99] to avoid invalid array accesses.
    if (sGL0 < 0.0001) sGL0 = 0.0001;
    if (sGL1 > 0.9999) sGL1 = 0.9999;

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
    const int pct_L = int(ceil(num_values * sGL0) + 0.001) - 1;
    const int pct_H = int(ceil(num_values * sGL1) + 0.001) - 1;
    const int median_ind = int(ceil(num_values * 0.5) + 0.001) - 1;

    // Compute the values that are +/- (width * sigmaG) from the median.
    float sigmaG = sigmaGCoeff * (values[idxArray[pct_H]]
                                  - values[idxArray[pct_L]]);
    float minValue = values[idxArray[median_ind]] - width * sigmaG;
    float maxValue = values[idxArray[median_ind]] + width * sigmaG;

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
