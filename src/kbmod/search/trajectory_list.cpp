#include "logging.h"

#include "trajectory_list.h"
#include "pydocs/trajectory_list_docs.h"

#include <parallel/algorithm>
#include <algorithm>

namespace search {

// -------------------------------------------------------
// --- Implementation of core data structure functions ---
// -------------------------------------------------------

TrajectoryList::TrajectoryList(uint64_t max_list_size) {
    max_size = max_list_size;

    // Start with the data on CPU.
    data_on_gpu = false;
    cpu_list.resize(max_size);
    gpu_array.resize(max_size);
}

TrajectoryList::TrajectoryList(const std::vector<Trajectory>& prev_list) {
    max_size = prev_list.size();
    cpu_list = prev_list;  // Do a full copy.

    // Start with the data on CPU.
    data_on_gpu = false;
    gpu_array.resize(max_size);
}

TrajectoryList::~TrajectoryList() {
    if (data_on_gpu) {
        logging::getLogger("kbmod.search.trajectory_list")
                ->debug("Freeing TrajectoryList on GPU. " + gpu_array.stats_string());
        gpu_array.free_gpu_memory();
    }
}

void TrajectoryList::resize(uint64_t new_size) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");

    cpu_list.resize(new_size);
    gpu_array.resize(new_size);
    max_size = new_size;
}

void TrajectoryList::set_trajectories(const std::vector<Trajectory>& new_values) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    const uint64_t new_size = new_values.size();

    resize(new_size);
    for (uint64_t i = 0; i < new_size; ++i) {
        cpu_list[i] = new_values[i];
    }
}

std::vector<Trajectory> TrajectoryList::get_batch(uint64_t start, uint64_t count) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    if (count == 0) throw std::runtime_error("count must be greater than 0");

    // If we are trying to read past the end of the array, just read until the end.
    if (start + count >= max_size) {
        return std::vector<Trajectory>(cpu_list.begin() + start, cpu_list.end());
    }
    return std::vector<Trajectory>(cpu_list.begin() + start, cpu_list.begin() + start + count);
}

void TrajectoryList::sort_by_likelihood() {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    __gnu_parallel::sort(cpu_list.begin(), cpu_list.end(),
                         [](const Trajectory& a, const Trajectory& b) { return b.lh < a.lh; });
}

void TrajectoryList::filter_by_likelihood(float min_lh) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");

    auto new_end = std::remove_if(cpu_list.begin(), cpu_list.end(),
                                  [min_lh](const Trajectory& a) { return (a.lh < min_lh); });
    cpu_list.erase(new_end, cpu_list.end());
    resize(cpu_list.size());
}

void TrajectoryList::filter_by_obs_count(int min_obs_count) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");

    auto new_end = std::remove_if(cpu_list.begin(), cpu_list.end(), [min_obs_count](const Trajectory& a) {
        return (a.obs_count < min_obs_count);
    });
    cpu_list.erase(new_end, cpu_list.end());
    resize(cpu_list.size());
}

void TrajectoryList::move_to_gpu() {
    if (data_on_gpu) return;  // Nothing to do.

    logging::getLogger("kbmod.search.trajectory_list")
            ->debug("Moving TrajectoryList to GPU. " + gpu_array.stats_string());

    // GPUArray handles all the validity checking, allocation, and copying.
    gpu_array.copy_vector_to_gpu(cpu_list);
    data_on_gpu = true;
}

void TrajectoryList::move_to_cpu() {
    if (!data_on_gpu) return;  // Nothing to do.

    logging::getLogger("kbmod.search.trajectory_list")
            ->debug("Freeing TrajectoryList on GPU: " + gpu_array.stats_string());

    // GPUArray handles all the validity checking and copying.
    gpu_array.copy_gpu_to_vector(cpu_list);
    gpu_array.free_gpu_memory();
    data_on_gpu = false;
}

// -------------------------------------------
// --- Helper functions ----------------------
// -------------------------------------------

std::vector<int> extract_all_trajectory_x(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<int> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].x;
    }
    return result;
}

std::vector<int> extract_all_trajectory_y(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<int> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].y;
    }
    return result;
}

std::vector<float> extract_all_trajectory_vx(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<float> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].vx;
    }
    return result;
}

std::vector<float> extract_all_trajectory_vy(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<float> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].vy;
    }
    return result;
}

std::vector<float> extract_all_trajectory_lh(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<float> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].lh;
    }
    return result;
}

std::vector<float> extract_all_trajectory_flux(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<float> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].flux;
    }
    return result;
}

std::vector<int> extract_all_trajectory_obs_count(const std::vector<Trajectory>& trajectories) {
    size_t num_trj = trajectories.size();

    std::vector<int> result(num_trj);
    for (size_t i = 0; i < num_trj; ++i) {
        result[i] = trajectories[i].obs_count;
    }
    return result;
}

// -------------------------------------------
// --- Python definitions --------------------
// -------------------------------------------

#ifdef Py_PYTHON_H
static void trajectory_list_binding(py::module& m) {
    using trjl = search::TrajectoryList;

    py::class_<trjl>(m, "TrajectoryList", pydocs::DOC_TrajectoryList)
            .def(py::init<int>())
            .def(py::init<std::vector<Trajectory>&>())
            .def_property_readonly("on_gpu", &trjl::on_gpu, pydocs::DOC_TrajectoryList_on_gpu)
            .def("__len__", &trjl::get_size)
            .def("resize", &trjl::resize, pydocs::DOC_TrajectoryList_resize)
            .def("get_size", &trjl::get_size, pydocs::DOC_TrajectoryList_get_size)
            .def("get_memory", &trjl::get_memory, pydocs::DOC_TrajectoryList_get_memory)
            .def("estimate_memory", &trjl::estimate_memory, pydocs::DOC_TrajectoryList_estimate_memory)
            .def("get_trajectory", &trjl::get_trajectory, py::return_value_policy::reference_internal,
                 pydocs::DOC_TrajectoryList_get_trajectory)
            .def("set_trajectory", &trjl::set_trajectory, pydocs::DOC_TrajectoryList_set_trajectory)
            .def("set_trajectories", &trjl::set_trajectories, pydocs::DOC_TrajectoryList_set_trajectories)
            .def("get_list", &trjl::get_list, pydocs::DOC_TrajectoryList_get_list)
            .def("get_batch", &trjl::get_batch, pydocs::DOC_TrajectoryList_get_batch)
            .def("sort_by_likelihood", &trjl::sort_by_likelihood,
                 pydocs::DOC_TrajectoryList_sort_by_likelihood)
            .def("filter_by_likelihood", &trjl::filter_by_likelihood,
                 pydocs::DOC_TrajectoryList_filter_by_likelihood)
            .def("filter_by_obs_count", &trjl::filter_by_obs_count,
                 pydocs::DOC_TrajectoryList_filter_by_obs_count)
            .def("move_to_cpu", &trjl::move_to_cpu, pydocs::DOC_TrajectoryList_move_to_cpu)
            .def("move_to_gpu", &trjl::move_to_gpu, pydocs::DOC_TrajectoryList_move_to_gpu);

    // Add the helper functions.
    m.def("extract_all_trajectory_x", &search::extract_all_trajectory_x,
          pydocs::DOC_TrajectoryList_extract_all_x);
    m.def("extract_all_trajectory_y", &search::extract_all_trajectory_y,
          pydocs::DOC_TrajectoryList_extract_all_y);
    m.def("extract_all_trajectory_vx", &search::extract_all_trajectory_vx,
          pydocs::DOC_TrajectoryList_extract_all_vx);
    m.def("extract_all_trajectory_vy", &search::extract_all_trajectory_vy,
          pydocs::DOC_TrajectoryList_extract_all_vy);
    m.def("extract_all_trajectory_lh", &search::extract_all_trajectory_lh,
          pydocs::DOC_TrajectoryList_extract_all_lh);
    m.def("extract_all_trajectory_flux", &search::extract_all_trajectory_flux,
          pydocs::DOC_TrajectoryList_extract_all_flux);
    m.def("extract_all_trajectory_obs_count", &search::extract_all_trajectory_obs_count,
          pydocs::DOC_TrajectoryList_extract_all_obs_count);
}
#endif

} /* namespace search */
