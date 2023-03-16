/*
 * TrajectoryUtils.cpp
 *
 * Created on: Sept 23, 2022
 *
 * Helper functions for filtering results.
 */

#include "TrajectoryUtils.h"

namespace search {

/* Compute the average distance between two trajectories at the
   given time steps. Used in duplicate filtering and clustering. */
double aveTrajectoryDistance(const std::vector<pixelPos>& posA, const std::vector<pixelPos>& posB) {
    const int num_times = posA.size();
    assert(posB.size() == num_times);

    double sum = 0.0;
    for (int i = 0; i < num_times; ++i) {
        double dx = posA[i].x - posB[i].x;
        double dy = posA[i].y - posB[i].y;
        sum += sqrt(dx * dx + dy * dy);
    }

    return sum / (double)num_times;
}

} /* namespace search */
