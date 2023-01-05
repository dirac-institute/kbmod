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

} /* namespace search */

#endif /* FILTERING_H_ */
