/*
 * Filtering.cpp
 *
 * Created on: Sept 2, 2022
 *
 * Helper functions for filtering results.
 */

#include "Filtering.h"
#include <math.h>

namespace kbmod {

/* The filter_kenerls.cu functions. */
extern "C" void sigmaGFilteredIndicesCU(float* values, int num_values, float sGL0, float sGL1,
                                        float sigmaGCoeff, float width, int* idxArray, int* minKeepIndex,
                                        int* maxKeepIndex);

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
    sigmaGFilteredIndicesCU(values_arr, num_values, sGL0, sGL1, sigmaGCoeff, width, idxArray, &minKeepIndex,
                            &maxKeepIndex);

    // Copy the result into a vector and return it.
    std::vector<int> result;
    for (int i = minKeepIndex; i <= maxKeepIndex; ++i) {
        result.push_back(idxArray[i]);
    }
    return result;
}

std::vector<int> clippedAverageFilteredIndices(const std::vector<float>& psi_curve,
                                               const std::vector<float>& phi_curve,
                                               int num_clipped, int n_sigma, float lower_lh_limit) {
    /*
     * This function applies a clipped median filter to a set of likelihood values. The largest
     * likelihood values (N=num_clipped) are eliminated if they are more than n_sigma*st_dev
     * away from the median, which is calculated excluding the largest values.
     *
     * Arguments:
     *     psi_curve - The psi values.
     *     phi_curve - The phi values.
     *     num_clipped - The number of likelihood values to consider eliminating. Only
     *                   considers the largest N=num_clipped values.
     *     n_sigma - The number of standard deviations away from the median that the largest
     *               likelihood values (N=num_clipped) must be in order to be eliminated.
     *     lower_lh_limit - Likelihood values lower than lower_lh_limit are automatically
     *                      eliminated from consideration.
     *
     * Returns: The indices of psi & phi that pass the filtering.
     */
    const int num_times = psi_curve.size();
    assert (num_times == phi_curve.size());

    // Compute the likelihood for each index.
    std::vector<std::pair<float, int> > scores;
    for (int t = 0; t < num_times; ++t) {
        float val = 0.0;
        if ((psi_curve[t] == NO_DATA) || (phi_curve[t] == NO_DATA)) {
            val = NO_DATA;
        } else if (phi_curve[t] > 0.0) {
            val = psi_curve[t] / sqrt(phi_curve[t]);
        }
        scores.push_back(std::make_pair(val, t));
    }
    sort(scores.begin(), scores.end());

    // Find all the indices where:
    // 1) LH > lower_lh_limit
    // 2) They do not have one of the top 'num_clipped' likelihoods.
    // 3) LH != NO_DATA
    int first_valid = 0;
    int last_valid = 0;
    double lh_sum_good = 0.0;
    for (int t = 0; t < num_times - num_clipped; ++t) {
        if (scores[t].first > lower_lh_limit && scores[t].first != NO_DATA) {
            last_valid = t;
            lh_sum_good += scores[t].first;
        } else {
            first_valid++;
        }
    }

    // If nothing passes the filter, do not bother with further filtering.
    int num_values = last_valid - first_valid + 1;
    if (num_values == 0) {
        std::vector<int> empty_vect;
        return empty_vect;
    }

    // Compute the median LH.
    double median_val = 0.0;
    if (num_values % 2 == 1) {
        median_val = scores[(last_valid + first_valid) / 2].first;
    } else {
        median_val = (scores[(last_valid + first_valid) / 2].first
                      + scores[(last_valid + first_valid) / 2 + 1].first) / 2.0;
    }

    // Compute the mean and variance of the non-filtered LHs.
    double mean = lh_sum_good / static_cast<float>(num_values);
    double sum_diff_sq = 0.0;
    for (int t = first_valid; t <= last_valid; ++t) {
        sum_diff_sq += (scores[t].first - mean) * (scores[t].first - mean);
    }
    double sigma = sqrt(sum_diff_sq / static_cast<double>(num_values));

    // Find all the indices where:
    // 1) LH > lower_lh_limit (by starting at first_valid)
    // 2) LH is below median + n_sigma * sigma
    // 3) LH != NO_DATA (by starting at first_valid)
    std::vector<int> passed_indices;
    for (int t = first_valid; t < num_times; ++t) {
        if (scores[t].first < median_val + sigma * static_cast<double>(n_sigma)) {
            passed_indices.push_back(scores[t].second);
        }
    }

    // Sort the indices
    sort(passed_indices.begin(), passed_indices.end());
    return passed_indices;
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

/* Given a set of fluxValues, inverse Phi values, and a vector of positive flux indicies,
   apply the kalman filters to said positive values.
   fluxValues: vector of flux values (psi / phi)
   vars: vector of inverse phi values
   fluxIdx: vector of valid fluxValues to be used in calculation
   reverse: whether or not the kalman filter should be calculated
   "forward" (0 -> end of vector) or "reversed" (end of vector -> 0) */
std::tuple<std::vector<double>, std::vector<double>> calculateKalmanFlux(std::vector<double> fluxValues,
                                                                         std::vector<double> vars,
                                                                         std::vector<int> fluxIdx,
                                                                         bool reverse) {
    int fluxSize = fluxValues.size();

    std::vector<double> xhat(fluxSize, 0.0);
    std::vector<double> p(fluxSize, 0.0);
    double xhatMinus = 0.0;
    double pMinus = 0.0;
    double k = 0.0;

    double q = 1.0;

    /* If reverse is false, calculate
       Kalman result going 'forward'
       (0 -> end of fluxes)
       Else, calculate it 'backwards'
       (end of fluxes -> 0)*/
    if (!reverse) {
        xhat[fluxIdx[0]] = fluxValues[fluxIdx[0]];
        p[fluxIdx[0]] = vars[fluxIdx[0]];

        for (int i = 1; i < fluxIdx.size(); i++) {
            int idx = fluxIdx[i];
            int idxMinus = fluxIdx[i - 1];

            xhatMinus = xhat[idxMinus];
            pMinus = p[idxMinus] + q;
            k = pMinus / (pMinus + vars[idx]);

            xhat[idx] = xhatMinus + k * (fluxValues[idx] - xhatMinus);
            p[idx] = (1.0 - k) * pMinus;
        }
    } else {
        xhat[fluxIdx.back()] = fluxValues[fluxIdx.back()];
        p[fluxIdx.back()] = vars[fluxIdx.back()];

        for (int j = fluxIdx.size() - 2; j >= 0; j--) {
            int idx = fluxIdx[j];
            int idxPlus = fluxIdx[j + 1];

            xhatMinus = xhat[idxPlus];
            pMinus = p[idxPlus] + q;
            k = pMinus / (pMinus + vars[idx]);

            xhat[idx] = xhatMinus + k * (fluxValues[idx] - xhatMinus);
            p[idx] = (1.0 - k) * pMinus;
        }
    }

    return std::make_tuple(xhat, p);
}

/* Apply kalman filtering to a single index
   and return the new likelihood */
std::tuple<std::vector<int>, double> kalmanFilterIndex(std::vector<double> psiCurve,
                                                       std::vector<double> phiCurve)

{
    int numValues = psiCurve.size();
    double maskVal = 1.0 / 9999999.0;
    std::vector<double> fluxValues(numValues, 0.0);
    std::vector<double> vars(numValues, maskVal);
    std::vector<int> fluxIdx;

    /* Return a set of curve indicies equal to
       { -1 } to indicate that this index should
       be filtered out. */
    std::vector<int> failIndex = {-1};
    auto filterIndex = std::make_tuple(failIndex, 0.0);

    // Mask phi values, calculate the flux,
    // the inverse phi values, and get a list
    // of positive flux values.
    for (int i = 0; i < numValues; i++) {
        double masked_phi = phiCurve[i];
        if (masked_phi == 0.0) {
            masked_phi = 1e9;
        }
        fluxValues[i] = psiCurve[i] / masked_phi;
        vars[i] = 1.0 / masked_phi;

        if (fluxValues[i] > 0.0) {
            fluxIdx.push_back(i);
        }
    }

    if (fluxIdx.size() < 2) {
        return filterIndex;
    }

    // First kalman filter pass
    auto kr1 = calculateKalmanFlux(fluxValues, vars, fluxIdx, false);

    std::vector<int> keepIdx1;

    for (int k = 0; k < fluxIdx.size(); k++) {
        double flux = std::get<0>(kr1)[fluxIdx[k]];
        double error = std::get<1>(kr1)[fluxIdx[k]];

        if (error < 0.0) {
            return filterIndex;
        }

        double deviation = abs(flux - fluxValues[fluxIdx[k]]) / pow(error, 0.5);
        if (deviation < 5.0) {
            keepIdx1.push_back(fluxIdx[k]);
        }
    }

    // Second kalman filter pass
    // (reversed in case first elements are extra bright)
    auto kr2 = calculateKalmanFlux(fluxValues, vars, fluxIdx, true);

    std::vector<int> keepIdx2;

    for (int l = 0; l < fluxIdx.size(); l++) {
        double flux = std::get<0>(kr2)[fluxIdx[l]];
        double error = std::get<1>(kr2)[fluxIdx[l]];

        if (error < 0.0) {
            return filterIndex;
        }

        double deviation = abs(flux - fluxValues[fluxIdx[l]]) / pow(error, 0.5);
        if (deviation < 5.0) {
            keepIdx2.push_back(fluxIdx[l]);
        }
    }

    // Choose which pass of results to use.
    std::vector<int> resultIdx;

    if (keepIdx1.size() >= keepIdx2.size()) {
        resultIdx = keepIdx1;
    } else {
        resultIdx = keepIdx2;
    }

    if (resultIdx.size() == 0) {
        return filterIndex;
    }

    // Calculate the new likelihood value
    // for the index.
    std::vector<double> newPsi;
    std::vector<double> newPhi;

    for (int m = 0; m < resultIdx.size(); m++) {
        newPsi.push_back(psiCurve[resultIdx[m]]);
        newPhi.push_back(phiCurve[resultIdx[m]]);
    }

    double newLikelihood = calculateLikelihoodFromPsiPhi(newPsi, newPhi);

    return std::make_tuple(resultIdx, newLikelihood);
}

/* Iterates over a given set of indices and runs them through the Kalman
   Filtering process.
*/
std::vector<std::tuple<int, std::vector<int>, double>> kalmanFiteredIndices(
        const std::vector<std::vector<double>>& psiValues,
        const std::vector<std::vector<double>>& phiValues) {
    std::vector<std::tuple<int, std::vector<int>, double>> kalmanIndices;
    int numValues = psiValues.size();
    for (int i = 0; i < numValues; i++) {
        auto result = kalmanFilterIndex(psiValues[i], phiValues[i]);
        kalmanIndices.push_back(std::make_tuple(i, std::get<0>(result), std::get<1>(result)));
    }
    return kalmanIndices;
}

} /* namespace kbmod */
