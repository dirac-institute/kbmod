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

void TrajectoryList::filter_by_likelihood(float min_likelihood) {
    sort_by_likelihood();

    // Find the first index that does not meet the threshold.
    uint64_t index = 0;
    while ((index < max_size) && (cpu_list[index].lh >= min_likelihood)) {
        ++index;
    }

    // Drop the values below the threshold.
    resize(index);
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
            .def("move_to_cpu", &trjl::move_to_cpu, pydocs::DOC_TrajectoryList_move_to_cpu)
            .def("move_to_gpu", &trjl::move_to_gpu, pydocs::DOC_TrajectoryList_move_to_gpu);
}
#endif

} /* namespace search */
