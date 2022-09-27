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
extern "C" void sigmaGFilteredIndicesCU(float* values, int num_values,
        float sGL0, float sGL1, float sigmaGCoeff, float width,
        int* idxArray, int* minKeepIndex, int* maxKeepIndex);


/* Return the list of indices from the values array such that those elements
   pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
   sigmaGCoeff and a multiplicative factor of width. */
std::vector<int> sigmaGFilteredIndices(const std::vector<float>& values,
                                       float sGL0, float sGL1, float sigmaGCoeff,
                                       float width)
{
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
    sigmaGFilteredIndicesCU(values_arr, num_values, sGL0, sGL1, sigmaGCoeff, width,
                            idxArray, &minKeepIndex, &maxKeepIndex);

    // Copy the result into a vector and return it.
    std::vector<int> result;
    for (int i = minKeepIndex; i <= maxKeepIndex; ++i) {
        result.push_back(idxArray[i]);
    }
    return result;
}
    
double calculateLikelihood(std::vector<double> psiValues, std::vector<double> phiValues)
{
    double psiSum = 0.0;
    double phiSum = 0.0;
    
    for(int i = 0; i < psiValues.size(); i++) {
        psiSum += psiValues[i];
        phiSum += phiValues[i];
        
    }
    
    if(psiSum == 0.0 || phiSum <= 0.0) {
        return 0.0;
    }
    
    return psiSum / sqrt(phiSum);
}
    
std::tuple<std::vector<double>, std::vector<double>> calculateKalmanFlux(std::vector<double> fluxValues, 
                                                                         std::vector<double> invFluxes,
                                                                         std::vector<int> fluxIdx, int pass)
{
    int fluxSize = fluxValues.size();
    double kalmanFluxes[fluxSize];
    
    std::vector<double> xhat(fluxSize, 0.0);
    std::vector<double> p(fluxSize, 0.0);
    double xhatMinus[fluxSize];
    double pMinus[fluxSize];
    double k[fluxSize];
    
    double q = 1.0;
    
    if(pass == 1) {
        xhat[fluxIdx[0]] = fluxValues[fluxIdx[0]];
        p[fluxIdx[0]] = invFluxes[fluxIdx[0]];
        
        for(int i = 1; i < fluxIdx.size(); i++) {
            int idx = fluxIdx[i];
            int idxMinus = fluxIdx[i - 1];
            
            xhatMinus[idx] = xhat[idxMinus];
            pMinus[idx] = p[idxMinus] + q;

            k[idx] = pMinus[idx] / (pMinus[idx] + invFluxes[idx]);
            xhat[idx] = xhatMinus[idx] + k[idx]*(fluxValues[idx]-xhatMinus[idx]);
            p[idx] = (1.0-k[idx])*pMinus[idx];
        }
    }
    else {
        xhat[fluxIdx.back()] = fluxValues[fluxIdx.back()];
        p[fluxIdx.back()] = invFluxes[fluxIdx.back()];
        
        for(int j = fluxIdx.size() - 2; j >= 0; j--) {
            int idx = fluxIdx[j];
            xhatMinus[idx] = xhat[fluxIdx[j + 1]];
            pMinus[idx] = p[fluxIdx[j + 1]] + q;

            k[idx] = pMinus[idx] / (pMinus[idx] + invFluxes[idx]);
            xhat[idx] = xhatMinus[idx] + k[idx]*(fluxValues[idx]-xhatMinus[idx]);
            p[idx] = (1.0-k[idx])*pMinus[idx];
        }
    }
    
    return std::make_tuple(xhat, p);
}
    
std::tuple<std::vector<int>, double> kalmanFilterIndex(std::vector<double> psiCurve,
                                                       std::vector<double> phiCurve)
    
{
    int numValues = psiCurve.size();
    double maskVal = 1.0 / 9999999.0;
    std::vector<double> fluxValues(numValues, 0.0);
    std::vector<double> invFluxes(numValues, maskVal);
    std::vector<int> fluxIdx;
    std::vector<int> failIndex = { -1 };
    
    for(int i = 0; i < numValues; i++) {
        double masked_phi = phiCurve[i];
        if(masked_phi == 0.0 ) {
            masked_phi = 1e9;
        }
        if(masked_phi < -999.0) {
            masked_phi = 9999999.0;
        }
        fluxValues[i] = psiCurve[i] / masked_phi;
        invFluxes[i] = 1.0 / masked_phi;
        
        if(fluxValues[i] > 0.0) {
            fluxIdx.push_back(i);
        }
    }
    
    int numPosFlux = fluxIdx.size();
    if(numPosFlux < 2) {
        return std::make_tuple(failIndex, 0.0);
    }
    
    auto kr1 = calculateKalmanFlux(fluxValues, invFluxes, fluxIdx, 1);
    
    std::vector<int> keepIdx1;
    
    for(int k = 0; k < fluxIdx.size(); k++) {
        double flux = std::get<0>(kr1)[fluxIdx[k]];
        double error = std::get<1>(kr1)[fluxIdx[k]];
        
        if(error < 0.0) {
            return std::make_tuple(failIndex, 0.0);
        }
        
        double deviation = abs(flux - fluxValues[fluxIdx[k]]) / pow(error, 0.5);
        if(deviation < 5.0) {
            keepIdx1.push_back(fluxIdx[k]);
        }
    }
    
    auto kr2 = calculateKalmanFlux(fluxValues, invFluxes, fluxIdx, 2);
    
    std::vector<int> keepIdx2;
    
    for(int l = 0; l < fluxIdx.size(); l++) {
        double flux = std::get<0>(kr2)[fluxIdx[l]];
        double error = std::get<1>(kr2)[fluxIdx[l]];
        
        if(error < 0.0) {
            return std::make_tuple(failIndex, 0.0);
        }
        
        double deviation = abs(flux - fluxValues[fluxIdx[l]]) / pow(error, 0.5);
        if(deviation < 5.0) {
            keepIdx2.push_back(fluxIdx[l]);
        }
    }
    
    std::vector<int> resultIdx = keepIdx1;
    
    if(keepIdx1.size() >= keepIdx2.size()) {
        resultIdx = keepIdx1;
    } else {
        resultIdx = keepIdx2;
    }
    
    if(resultIdx.size() == 0) {
        return std::make_tuple(failIndex, 0.0);
    }
    
    std::vector<double> newPsi;
    std::vector<double> newPhi;

    for(int m = 0; m < resultIdx.size(); m++) {
        newPsi.push_back(psiCurve[resultIdx[m]]);
        newPhi.push_back(phiCurve[resultIdx[m]]);
    }
    
    double newLikelihood = calculateLikelihood(newPsi, newPhi);
    
    return std::make_tuple(resultIdx, newLikelihood);
}
                                  
    
std::vector<std::tuple<int, std::vector<int>, double>> kalmanFiteredIndices(const std::vector<std::vector<double>>& psiValues, 
                                                                            const std::vector<std::vector<double>>& phiValues,
                                                                            int numValues)
{
    std::vector<std::tuple<int, std::vector<int>, double>> kalmanIndices;
    for(int i = 0; i < numValues; i++) {
        auto result = kalmanFilterIndex(psiValues[i], phiValues[i]);
        kalmanIndices.push_back(std::make_tuple(i, std::get<0>(result), std::get<1>(result)));
    }
    return kalmanIndices;
}
    
} /* namespace kbmod */
