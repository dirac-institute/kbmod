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

namespace search {

/* Return the list of indices from the values array such that those elements
   pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
   sigmaGCoeff and a multiplicative factor of width. */
std::vector<int> sigmaGFilteredIndices(const std::vector<float>& values, float sGL0, float sGL1,
                                       float sigmaGCoeff, float width);

// This function applies a clipped median filter to a set of likelihood values. The largest
// likelihood values (N=num_clipped) are eliminated if they are more than n_sigma*st_dev
// away from the median, which is calculated excluding the largest values.
std::vector<int> clippedAverageFilteredIndices(const std::vector<float>& psi_curve,
                                               const std::vector<float>& phi_curve, int num_clipped,
                                               int n_sigma, float lower_lh_limit);

double calculateLikelihoodFromPsiPhi(std::vector<double> psiValues, std::vector<double> phiValues);

std::tuple<std::vector<double>, std::vector<double>> calculateKalmanFlux(std::vector<double> fluxValues,
                                                                         std::vector<double> vars,
                                                                         std::vector<int> fluxIdx,
                                                                         bool reverse);

std::tuple<std::vector<int>, double> kalmanFilterIndex(std::vector<double> psiCurve,
                                                       std::vector<double> phiCurve);

std::vector<std::tuple<int, std::vector<int>, double>> kalmanFiteredIndices(
        const std::vector<std::vector<double>>& psiValues, const std::vector<std::vector<double>>& phiValues);
} /* namespace search */

#endif /* FILTERING_H_ */
