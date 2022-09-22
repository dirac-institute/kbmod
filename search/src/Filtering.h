/*
 * Filtering.h
 *
 * Created on: Sept 2, 2022
 *
 * Helper functions for filtering results.
 */

#ifndef FILTERING_H_
#define FILTERING_H_

#include <vector>

namespace kbmod {

/* Return the list of indices from the values array such that those elements
   pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
   sigmaGCoeff and a multiplicative factor of width. */
std::vector<int> sigmaGFilteredIndices(const std::vector<float>& values,
                                       float sGL0, float sGL1,
                                       float sigmaGCoeff, float width);
    
float calculateLikelihood(std::vector<float> psiValues, std::vector<float> phiValues);
    
std::tuple<std::vector<float>, std::vector<float>> calculateKalmanFlux(std::vector<float> fluxValues, 
                                                                       std::vector<float> invFluxes,
                                                                       std::vector<int> fluxIdx, int pass);
    
std::tuple<std::vector<int>, float> kalmanFilterIndex(std::vector<float>& psiCurve,
                                                      std::vector<float>& phiCurve);
    
std::vector<std::tuple<int, std::vector<int>, float>> kalmanFiteredIndices(const std::vector<std::vector<float>>& psiValues, 
                                                                           const std::vector<std::vector<float>>& phiValues,
                                                                           int numValues);

} /* namespace kbmod */

#endif /* FILTERING_H_ */
