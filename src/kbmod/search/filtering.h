#ifndef FILTERING_H_
#define FILTERING_H_

#include <vector>


namespace search {
  /* Return the list of indices from the values array such that those elements
     pass the sigmaG filtering defined by percentiles [sGL0, sGL1] with coefficient
     sigmag_coeff and a multiplicative factor of width. */
  std::vector<int> sigmaGFilteredIndices(const std::vector<float>& values, float sgl0, float sgl1,
                                         float sigma_g_coeff, float width);

} /* namespace search */

#endif /* FILTERING_H_ */
