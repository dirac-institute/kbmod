/*
 * TrajectoryUtils.cpp
 *
 * Created on: Sept 23, 2022
 *
 * Helper functions for filtering results.
 */

#include "TrajectoryUtils.h"

namespace search {

TrajectoryResult::TrajectoryResult(const trajectory& trj, int num_times) : trj_(trj), num_times_(num_times) {
    valid_indices_.resize(num_times, true);
}

TrajectoryResult::TrajectoryResult(const trajectory& trj, const std::vector<int>& binary_valid) : trj_(trj) {
    num_times_ = binary_valid.size();
    valid_indices_.resize(num_times_, true);
    for (int i = 0; i < num_times_; ++i) {
        valid_indices_[i] = (binary_valid[i] != 0);
    }
}

TrajectoryResult::TrajectoryResult(const trajectory& trj, int num_times,
                                   const std::vector<int>& valid_indices)
        : trj_(trj), num_times_(num_times) {
    const int num_valid = valid_indices.size();
    valid_indices_.resize(num_times, false);
    for (int i = 0; i < num_valid; ++i) {
        int ind = valid_indices[i];
        assert((ind >= 0) && (ind < num_times_));
        valid_indices_[ind] = true;
    }
}

inline bool TrajectoryResult::check_index_valid(int index) const {
    return ((index >= 0) && (index < num_times_)) ? valid_indices_[index] : false;
}

inline void TrajectoryResult::set_index_valid(int index, bool is_valid) {
    if ((index >= 0) && (index < num_times_)) {
        valid_indices_[index] = is_valid;
    }
}

std::vector<int> TrajectoryResult::get_valid_indices_list() const {
    std::vector<int> inds;
    for (int i = 0; i < num_times_; ++i) {
        if (valid_indices_[i]) inds.push_back(i);
    }
    return inds;
}

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
