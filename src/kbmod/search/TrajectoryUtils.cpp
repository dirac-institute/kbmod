/*
 * TrajectoryUtils.cpp
 *
 * Created on: Sept 23, 2022
 *
 * Helper functions for filtering results.
 */

#include "TrajectoryUtils.h"

namespace search {

TrajectoryResult::TrajectoryResult(const trajectory& trj, int num_times) :
        trj_(trj), num_times_(num_times) {
    valid_indices_.resize(num_times, true);
}

TrajectoryResult::TrajectoryResult(const trajectory& trj, const std::vector<int>& binary_valid) : trj_(trj) {
    num_times_ = binary_valid.size();
    valid_indices_.reserve(num_times_);
    for (int i = 0; i < num_times_; ++i) {
        valid_indices_[i] = (binary_valid[i] != 0);
    }
}

TrajectoryResult::TrajectoryResult(const trajectory& trj, int num_times, const std::vector<int>& valid_indices) :
        trj_(trj), num_times_(num_times) {
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

// Converts a trajRegion result into a trajectory result.
trajectory convertTrajRegion(const trajRegion& t, float endTime) {
    assert(endTime > 0.0);
    float scale = 1.0;
    if (t.depth > 0) scale = std::pow(2.0, static_cast<float>(t.depth));

    trajectory tb;
    tb.lh = t.likelihood;
    tb.flux = t.flux;
    tb.obsCount = t.obs_count;
    tb.x = t.ix * scale;
    tb.y = t.iy * scale;
    tb.xVel = (scale * t.fx - scale * t.ix) / endTime;
    tb.yVel = (scale * t.fy - scale * t.iy) / endTime;
    return tb;
}

std::vector<trajRegion> subdivideTrajRegion(const trajRegion& t) {
    short nDepth = t.depth - 1;
    std::vector<trajRegion> children(16);
    float nix = t.ix * 2.0;
    float niy = t.iy * 2.0;
    float nfx = t.fx * 2.0;
    float nfy = t.fy * 2.0;
    const float s = 1.0;
    children[0] = {nix, niy, nfx, nfy, nDepth, 0, 0.0, 0.0};
    children[1] = {nix + s, niy, nfx, nfy, nDepth, 0, 0.0, 0.0};
    children[2] = {nix, niy + s, nfx, nfy, nDepth, 0, 0.0, 0.0};
    children[3] = {nix + s, niy + s, nfx, nfy, nDepth, 0, 0.0, 0.0};
    children[4] = {nix, niy, nfx + s, nfy, nDepth, 0, 0.0, 0.0};
    children[5] = {nix + s, niy, nfx + s, nfy, nDepth, 0, 0.0, 0.0};
    children[6] = {nix, niy + s, nfx + s, nfy, nDepth, 0, 0.0, 0.0};
    children[7] = {nix + s, niy + s, nfx + s, nfy, nDepth, 0, 0.0, 0.0};
    children[8] = {nix, niy, nfx, nfy + s, nDepth, 0, 0.0, 0.0};
    children[9] = {nix + s, niy, nfx, nfy + s, nDepth, 0, 0.0, 0.0};
    children[10] = {nix, niy + s, nfx, nfy + s, nDepth, 0, 0.0, 0.0};
    children[11] = {nix + s, niy + s, nfx, nfy + s, nDepth, 0, 0.0, 0.0};
    children[12] = {nix, niy, nfx + s, nfy + s, nDepth, 0, 0.0, 0.0};
    children[13] = {nix + s, niy, nfx + s, nfy + s, nDepth, 0, 0.0, 0.0};
    children[14] = {nix, niy + s, nfx + s, nfy + s, nDepth, 0, 0.0, 0.0};
    children[15] = {nix + s, niy + s, nfx + s, nfy + s, nDepth, 0, 0.0, 0.0};

    return children;
}

std::vector<trajRegion>& filterTrajRegionsLH(std::vector<trajRegion>& tlist, float minLH, int minObs) {
    tlist.erase(std::remove_if(tlist.begin(), tlist.end(),
                               std::bind([](trajRegion t, int mObs,
                                            float mLH) { return t.obs_count < mObs || t.likelihood < mLH; },
                                         std::placeholders::_1, minObs, minLH)),
                tlist.end());
    return tlist;
}

} /* namespace search */
