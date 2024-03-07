/*
 * trajectory_list.h
 *
 * The data structure for the raw results (a list of trajectories). The
 * data structure handles memory allocation on both the CPU and GPU and
 * transfering data between the two.
 *
 * Created on: March 7, 2024
 */

#ifndef TRAJECTORY_LIST_DS_
#define TRAJECTORY_LIST_DS_

#include <vector>

#include "common.h"

namespace search {

class TrajectoryList {
public:
    explicit TrajectoryList(int max_list_size);
    explicit TrajectoryList(const std::vector<Trajectory>& prev_list);
    virtual ~TrajectoryList();

    // --- Getter functions ----------------
    inline int get_size() const { return max_size; }

    inline Trajectory& get_trajectory(int index) {
        if (index < 0 || index > max_size) throw std::runtime_error("Index out of bounds.");
        if (data_on_gpu) throw std::runtime_error("Data on GPU");
        return cpu_list[index];
    }

    inline void set_trajectory(int index, const Trajectory& new_value) {
        if (index < 0 || index > max_size) throw std::runtime_error("Index out of bounds.");
        if (data_on_gpu) throw std::runtime_error("Data on GPU");
        cpu_list[index] = new_value;
    }

    inline std::vector<Trajectory>& get_list() {
        if (data_on_gpu) throw std::runtime_error("Data on GPU");
        return cpu_list;
    }

    // Forcibly resize. May add blank data.
    void resize(int new_size);

    // Get a batch of results.
    std::vector<Trajectory> get_batch(int start, int count);

    // Processing functions for sorting or filtering.
    void sort_by_likelihood();
    void sort_by_obs_count();
    void filter_by_likelihood(float min_likelihood);
    void filter_by_obs_count(int min_obs_count);

    // Data allocation functions.
    inline bool on_gpu() const { return data_on_gpu; }
    void move_to_gpu();
    void move_to_cpu();

    // Array access functions. For use when passing to the GPU only.
    inline Trajectory* get_gpu_list_ptr() { return gpu_list_ptr; }

private:
    int max_size;
    bool data_on_gpu;

    std::vector<Trajectory> cpu_list;
    Trajectory* gpu_list_ptr = nullptr;
};

} /* namespace search */

#endif /* TRAJECTORY_LIST_DS_ */
