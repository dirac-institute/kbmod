/*
 * TrajectoryUtils.h
 *
 * Created on: Sept 23, 2022
 *
 * Helper functions for trajectories and trajRegions.
 */

#ifndef TRAJECTORYUTILS_H_
#define TRAJECTORYUTILS_H_

#include "common.h"
#include <cmath>
#include <vector>

namespace search {

/* TrajectoryResult provides a wrapper for results that can be passed to filtering functions. */
class TrajectoryResult {
public:
    // Default all indices to valid.
    TrajectoryResult(const trajectory& trj, int num_times);

    // Take in a binary array indicating if each element is valid (1) or invalid (0).
    TrajectoryResult(const trajectory& trj, const std::vector<int>& binary_valid);

    // Take in an array of the individual indices that are valid.
    TrajectoryResult(const trajectory& trj, int num_times, const std::vector<int>& valid_indices);

    virtual ~TrajectoryResult(){};

    // Simple inline getters.
    trajectory& get_trajectory() { return trj_; }
    const trajectory& get_const_trajectory() const { return trj_; }
    int num_times() const { return num_times_; }
    bool check_index_valid(int index) const;

    // Get the list of indices that are valid. Takes linear time.
    std::vector<int> get_valid_indices_list() const;

    // Simple inline setters.
    void set_index_valid(int index, bool is_valid);

private:
    trajectory trj_;
    int num_times_;
    std::vector<bool> valid_indices_;
};

/* Compute the predicted trajectory position. */
inline pixelPos computeTrajPos(const trajectory& t, float time) {
    return {t.x + time * t.xVel, t.y + time * t.yVel};
}

inline pixelPos computeTrajPosBC(const trajectory& t, float time, const baryCorrection& bc) {
    return {t.x + time * t.xVel + bc.dx + t.x * bc.dxdx + t.y * bc.dxdy,
            t.y + time * t.yVel + bc.dy + t.x * bc.dydx + t.y * bc.dydy};
}

/* Compute the average distance between two trajectory's predicted
   positions. Used in duplicate filtering and clustering. */
double avePixelDistance(const std::vector<pixelPos>& posA, const std::vector<pixelPos>& posB);

/* --- Helper functions for trajRegion --------------- */

// Converts a trajRegion result into a trajectory result.
trajectory convertTrajRegion(const trajRegion& t, float endTime);

// Subdivide the trajRegion into 16 children at the next depth.
std::vector<trajRegion> subdivideTrajRegion(const trajRegion& t);

// Filter a vector of trajRegion to remove elements that do not
// have enough observations or likelihood.
std::vector<trajRegion>& filterTrajRegionsLH(std::vector<trajRegion>& tlist, float minLH, int minObs);
} /* namespace search */

#endif /* TRAJECTORYUTILS_H_ */
