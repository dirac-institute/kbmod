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
    
float calculateLikelihood(std::vector<float> psiValues, std::vector<float> phiValues)
{
    float psiSum;
    float phiSum;
    
    for(int i = 0; i < psiValues.size(); i++) {
        psiSum += psiValues[i];
        phiSum += phiValues[i];
        
    }
    
//     if(phiSum <= 0.0) {
//         return 0.0;
//     }
    
    return psiSum / sqrt(phiSum);
}
    
std::tuple<std::vector<float>, std::vector<float>> calculateKalmanFlux(std::vector<float> fluxValues, 
                                                                       std::vector<float> invFluxes,
                                                                       std::vector<int> fluxIdx, int pass)
{
    int fluxSize = fluxValues.size();
    float kalmanFluxes[fluxSize];
    
    std::vector<float> xhat(fluxSize, 0.0);
    std::vector<float> p(fluxSize, 0.0);
    float xhatMinus[fluxSize];
    float pMinus[fluxSize];
    float k[fluxSize];
    
    float q = 1.0;
    
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
    
std::tuple<std::vector<int>, float> kalmanFilterIndex(std::vector<float> psiCurve,
                                                      std::vector<float> phiCurve)
    
{
    int numValues = psiCurve.size();
    float maskVal = 1.0 / 9999999.0;
    std::vector<float> fluxValues(numValues, 0.0);
    std::vector<float> invFluxes(numValues, maskVal);
    std::vector<int> fluxIdx;
    std::vector<int> failIndex = { -1 };
    
    for(int i = 0; i < numValues; i++) {
        float masked_phi = phiCurve[i];
        if(masked_phi == 0.0 ) {
            masked_phi = 1e9;
        }
        fluxValues[i] = psiCurve[i] / masked_phi;
        
        if(fluxValues[i] > 0.0) {
            fluxIdx.push_back(i);
        }
    }
    
    int numPosFlux = fluxIdx.size();
    if(numPosFlux < 2) {
        return std::make_tuple(failIndex, 0.1);
    }
    
    for(int j = 0; j < numPosFlux; j++) {
        invFluxes[fluxIdx[j]] = 1.0 / fluxValues[fluxIdx[j]];
    } 
    
    auto kr1 = calculateKalmanFlux(fluxValues, invFluxes, fluxIdx, 1);
    
    std::vector<int> keepIdx1;
    float errorMin;
    
    for(int k = 0; k < fluxIdx.size(); k++) {
        float flux = std::get<0>(kr1)[fluxIdx[k]];
        float error = std::get<1>(kr1)[fluxIdx[k]];
        
        if(error < 0.0) {
            return std::make_tuple(failIndex, error);
        }
        
        float deviation = abs(flux - fluxValues[fluxIdx[k]]) / pow(error, 0.5);
        if(deviation < 5.0) {
            keepIdx1.push_back(fluxIdx[k]);
        }
    }
    
    auto kr2 = calculateKalmanFlux(fluxValues, invFluxes, fluxIdx, 2);
    
    std::vector<int> keepIdx2;
    
    for(int l = 0; l < fluxIdx.size(); l++) {
        float flux = std::get<0>(kr2)[fluxIdx[l]];
        float error = std::get<1>(kr2)[fluxIdx[l]];
        
        if(error < 0.0) {
            return std::make_tuple(failIndex, 0.3);
        }
        
        float deviation = abs(flux - fluxValues[fluxIdx[l]]) / pow(error, 0.5);
        if(deviation < 5.0) {
            keepIdx2.push_back(fluxIdx[l]);
        }
    }
    
    std::vector<int> resultIdx;
    
    if(keepIdx1.size() > keepIdx2.size()) {
        resultIdx = keepIdx1;
    } else {
        resultIdx = keepIdx2;
    }
    
    if(resultIdx.size() == 0) {
        return std::make_tuple(failIndex, 0.4);
    }
    
    std::vector<float> newPsi;
    std::vector<float> newPhi;

    for(int m = 0; m < resultIdx.size(); m++) {
        newPsi.push_back(psiCurve[resultIdx[m]]);
        newPhi.push_back(phiCurve[resultIdx[m]]);
    }
    
    float newLikelihood = calculateLikelihood(newPsi, newPhi);
    
    return std::make_tuple(resultIdx, newLikelihood);
}
                                  
    
std::vector<std::tuple<int, std::vector<int>, float>> kalmanFiteredIndices(const std::vector<std::vector<float>>& psiValues, 
                                                                           const std::vector<std::vector<float>>& phiValues,
                                                                           int numValues)
{
    std::vector<std::tuple<int, std::vector<int>, float>> kalmanIndices;
    for(int i = 0; i < numValues; i++) {
        auto result = kalmanFilterIndex(psiValues[i], phiValues[i]);
        kalmanIndices.push_back(std::make_tuple(i, std::get<0>(result), std::get<1>(result)));
    }
    return kalmanIndices;
}
    
} /* namespace kbmod */
