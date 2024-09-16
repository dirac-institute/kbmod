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
#include "gpu_array.h"

namespace search {

class TrajectoryList {
public:
    explicit TrajectoryList(uint64_t max_list_size);
    explicit TrajectoryList(const std::vector<Trajectory>& prev_list);
    virtual ~TrajectoryList();

    // Do not allow copy or assignment because that could interfere with the pointer deallocation.
    TrajectoryList(TrajectoryList&) = delete;
    TrajectoryList(const TrajectoryList&) = delete;
    TrajectoryList& operator=(TrajectoryList&) = delete;
    TrajectoryList& operator=(const TrajectoryList&) = delete;

    // --- Getter functions ----------------
    inline uint64_t get_size() const { return max_size; }

    inline Trajectory& get_trajectory(uint64_t index) {
        if (index >= max_size) throw std::runtime_error("Index out of bounds.");
        if (data_on_gpu) throw std::runtime_error("Data on GPU");
        return cpu_list[index];
    }

    inline void set_trajectory(uint64_t index, const Trajectory& new_value) {
        if (index >= max_size) throw std::runtime_error("Index out of bounds.");
        if (data_on_gpu) throw std::runtime_error("Data on GPU");
        cpu_list[index] = new_value;
    }

    void set_trajectories(const std::vector<Trajectory>& new_values);

    inline std::vector<Trajectory>& get_list() {
        if (data_on_gpu) throw std::runtime_error("Data on GPU");
        return cpu_list;
    }

    // Forcibly resize. May add blank data.
    void resize(uint64_t new_size);

    // Get a batch of results.
    std::vector<Trajectory> get_batch(uint64_t start, uint64_t count);

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
    inline Trajectory* get_gpu_list_ptr() { return gpu_array.get_ptr(); }

private:
    uint64_t max_size;
    bool data_on_gpu;

    std::vector<Trajectory> cpu_list;
    GPUArray<Trajectory> gpu_array;
};

} /* namespace search */

#endif /* TRAJECTORY_LIST_DS_ */
