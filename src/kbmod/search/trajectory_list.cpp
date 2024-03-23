#include "trajectory_list.h"
#include "pydocs/trajectory_list_docs.h"

#include <parallel/algorithm>
#include <algorithm>

namespace search {

// -------------------------------------------------------
// --- Implementation of core data structure functions ---
// -------------------------------------------------------

TrajectoryList::TrajectoryList(int max_list_size) {
    if (max_list_size < 0) {
        throw std::runtime_error("Invalid TrajectoryList size.");
    }
    max_size = max_list_size;

    // Start with the data on CPU.
    data_on_gpu = false;
    cpu_list.resize(max_size);
    gpu_array.resize(max_size);
}

// Move assignment operator.
TrajectoryList& TrajectoryList::operator=(TrajectoryList&& other) noexcept {
    if (this != &other) {
        max_size = other.max_size;
        data_on_gpu = other.data_on_gpu;
        cpu_list = std::move(other.cpu_list);
        gpu_array = std::move(other.gpu_array);
        other.data_on_gpu = false;
    }
    return *this;
}

TrajectoryList::TrajectoryList(const std::vector<Trajectory> &prev_list) {
    max_size = prev_list.size();
    cpu_list = prev_list;  // Do a full copy.

    // Start with the data on CPU.
    data_on_gpu = false;
    gpu_array.resize(max_size);
}

TrajectoryList::~TrajectoryList() {
    if (data_on_gpu) {
        gpu_array.free_gpu_memory();
    }
}

void TrajectoryList::resize(int new_size) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    if (new_size < 0) {
        throw std::runtime_error("Invalid TrajectoryList size.");
    }

    cpu_list.resize(new_size);
    gpu_array.resize(new_size);
    max_size = new_size;
}

void TrajectoryList::set_trajectories(const std::vector<Trajectory> &new_values) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    const int new_size = new_values.size();

    resize(new_size);
    for (int i = 0; i < new_size; ++i) {
        cpu_list[i] = new_values[i];
    }
}

std::vector<Trajectory> TrajectoryList::get_batch(int start, int count) {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    if (start < 0) throw std::runtime_error("start must be 0 or greater");
    if (count <= 0) throw std::runtime_error("count must be greater than 0");

    if (start + count >= max_size) {
        count = max_size - start;
    }
    return std::vector<Trajectory>(cpu_list.begin() + start, cpu_list.begin() + start + count);
}

void TrajectoryList::sort_by_likelihood() {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    __gnu_parallel::sort(cpu_list.begin(), cpu_list.end(),
                         [](Trajectory a, Trajectory b) { return b.lh < a.lh; });
}

void TrajectoryList::sort_by_obs_count() {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");
    __gnu_parallel::sort(cpu_list.begin(), cpu_list.end(),
                         [](Trajectory a, Trajectory b) { return b.obs_count < a.obs_count; });
}

void TrajectoryList::filter_by_likelihood(float min_likelihood) {
    sort_by_likelihood();

    // Find the first index that does not meet the threshold.
    int index = 0;
    while ((index < max_size) && (cpu_list[index].lh >= min_likelihood)) {
        ++index;
    }

    // Drop the values below the threshold.
    resize(index);
}

void TrajectoryList::filter_by_obs_count(int min_obs_count) {
    sort_by_obs_count();

    // Find the first index that does not meet the threshold.
    int index = 0;
    while ((index < max_size) && (cpu_list[index].obs_count >= min_obs_count)) {
        ++index;
    }

    // Drop the values below the threshold.
    resize(index);
}

void TrajectoryList::filter_by_valid() {
    if (data_on_gpu) throw std::runtime_error("Data on GPU");

    auto new_end = std::partition(cpu_list.begin(), cpu_list.end(), [](Trajectory x) { return x.valid; });
    int new_size = std::distance(cpu_list.begin(), new_end);
    resize(new_size);
}

void TrajectoryList::move_to_gpu() {
    if (data_on_gpu) return;  // Nothing to do.

    // GPUArray handles all the validity checking, allocation, and copying.
    gpu_array.copy_vector_to_gpu(cpu_list);
    data_on_gpu = true;
}

void TrajectoryList::move_to_cpu() {
    if (!data_on_gpu) return;  // Nothing to do.

    // GPUArray handles all the validity checking and copying.
    gpu_array.copy_gpu_to_vector(cpu_list);
    gpu_array.free_gpu_memory();
    data_on_gpu = false;
}

// -------------------------------------------
// --- Python definitions --------------------
// -------------------------------------------

#ifdef Py_PYTHON_H
static void trajectory_list_binding(py::module &m) {
    using trjl = search::TrajectoryList;

    py::class_<trjl>(m, "TrajectoryList", pydocs::DOC_TrajectoryList)
            .def(py::init<int>())
            .def(py::init<std::vector<Trajectory> &>())
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
            .def("sort_by_obs_count", &trjl::sort_by_obs_count, pydocs::DOC_TrajectoryList_sort_by_obs_count)
            .def("filter_by_likelihood", &trjl::filter_by_likelihood,
                 pydocs::DOC_TrajectoryList_filter_by_likelihood)
            .def("filter_by_obs_count", &trjl::filter_by_obs_count,
                 pydocs::DOC_TrajectoryList_filter_by_obs_count)
            .def("filter_by_valid", &trjl::filter_by_valid, pydocs::DOC_TrajectoryList_filter_by_valid)
            .def("move_to_cpu", &trjl::move_to_cpu, pydocs::DOC_TrajectoryList_move_to_cpu)
            .def("move_to_gpu", &trjl::move_to_gpu, pydocs::DOC_TrajectoryList_move_to_gpu);
}
#endif

} /* namespace search */
