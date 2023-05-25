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
double aveTrajectoryDistance(const std::vector<pixelPos>& posA, const std::vector<pixelPos>& posB);

} /* namespace search */

#endif /* TRAJECTORYUTILS_H_ */
