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
void evaluate_trajectory_cpu(const PsiPhiArray &psi_phi, Trajectory &candidate);

/*
 * Evaluate all of the candidate trajectories from a single starting pixel
 * (y, x) and return the best "num_results".
 *
 */
std::vector<Trajectory> evaluate_single_pixel(int y, int x, const PsiPhiArray &psi_phi,
                                              TrajectoryList &trj_to_search, int num_results);

/*
 * Perform the core KBMOD search (without sigma-G filtering) on CPU.
 */
void search_cpu_only(PsiPhiArray &psi_phi_array, SearchParameters params, TrajectoryList &trj_to_search,
                     TrajectoryList &results);

} /* namespace search */

#endif /* CPU_SEARCH_ALGOROTHMS_H_ */
