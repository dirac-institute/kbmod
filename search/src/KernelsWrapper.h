/*
 * KernelsWrapper.h
 *
 * Created on: Sept 2, 2022
 *
 * An interface to access some helper functions in the
 * kernels.cu file for better testing.
 */

#ifndef KERNELSWRAPPER_H_
#define KERNELSWRAPPER_H_

#include <vector>

namespace kbmod {

/* Return the list of indices from the values array such that those elements
   pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
   sigmaGCoeff */
std::vector<int> deviceSigmaGFilteredIndices(const std::vector<float>& values,
                                             float sGL0, float sGL1, float sigmaGCoeff);

} /* namespace kbmod */

#endif /* KERNELSWRAPPER_H_ */
