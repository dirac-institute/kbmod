/*
 * cpu_search_algorithms.cpp
 *
 * A purely CPU version of the critical kernel functions. While the __host__ tag
 * should be able to compile for CPU we fail if we do not have the nvidia driver
 * and libraries installed.
 */

#ifndef CPU_SEARCH_ALGOROTHMS_H_
#define CPU_SEARCH_ALGOROTHMS_H_

#include <cmath>
#include <vector>

#include "common.h"
#include "psi_phi_array_ds.h"
#include "trajectory_list.h"

namespace search {

/*
 * Evaluate the likelihood score (as computed with from the psi and phi values) for a single
 * given candidate trajectory. Modifies the trajectory in place to update the number of
 * observations, likelihood, and flux.
 *
 * Does not do sigma-G filtering.
 */
void evaluate_trajectory_cpu(PsiPhiArray& psi_phi, Trajectory& candidate);

} /* namespace search */

#endif /* CPU_SEARCH_ALGOROTHMS_H_ */
