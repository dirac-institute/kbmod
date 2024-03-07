#include "trajectory_list.h"
#include "pydocs/trajectory_list_docs.h"

#include <parallel/algorithm>
#include <algorithm>

namespace search {

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
extern "C" Trajectory *allocate_gpu_trajectory_list(long unsigned num_trj);

extern "C" void free_gpu_trajectory_list(Trajectory *gpu_ptr);

extern "C" void copy_trajectory_list(Trajectory *cpu_ptr, Trajectory *gpu_ptr, long unsigned num_trj,
                                     bool to_gpu);
#endif

// -------------------------------------------------------
// --- Implementation of core data structure functions ---
// -------------------------------------------------------

TrajectoryList::TrajectoryList(int max_list_size) {
    if (max_list_size <= 0) {
        throw std::runtime_error("Invalid TrajectoryList size.");
    }
    max_size = max_list_size;

    // Start with the data on CPU.
    data_on_gpu = false;
    cpu_list.resize(max_size);
    gpu_list_ptr = nullptr;
}

TrajectoryList::~TrajectoryList() {
#ifdef HAVE_CUDA
    if (gpu_list_ptr != nullptr) {
        free_gpu_trajectory_list(gpu_list_ptr);
        gpu_list_ptr = nullptr;
    }
#endif
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

void TrajectoryList::move_to_gpu() {
    if (data_on_gpu) return;  // Nothing to do.

    // Error checking.
    if (gpu_list_ptr != nullptr) throw std::runtime_error("GPU data already allocated.");
    if (cpu_list.size() != max_size) throw std::runtime_error("List size mismatch.");

        // Allocate the GPU array and copy the data onto GPU.
#ifdef HAVE_CUDA
    gpu_list_ptr = allocate_gpu_trajectory_list(max_size);
    copy_trajectory_list(cpu_list.data(), gpu_list_ptr, max_size, true);
#else
    throw std::runtime_error("CUDA installation to move things on or off GPU.");
#endif
    data_on_gpu = true;
}

void TrajectoryList::move_to_cpu() {
    if (!data_on_gpu) return;  // Nothing to do.

    // Error checking.
    if (gpu_list_ptr == nullptr) throw std::runtime_error("No GPU data allocated.");
    if (cpu_list.size() != max_size) throw std::runtime_error("List size mismatch.");

        // Copy the data to CPU and free the GPU array.
#ifdef HAVE_CUDA
    copy_trajectory_list(cpu_list.data(), gpu_list_ptr, max_size, false);
    free_gpu_trajectory_list(gpu_list_ptr);
    gpu_list_ptr = nullptr;
#else
    throw std::runtime_error("CUDA installation to move things on or off GPU.");
#endif

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
            .def_property_readonly("on_gpu", &trjl::on_gpu, pydocs::DOC_TrajectoryList_on_gpu)
            .def("__len__", &trjl::get_size)
            .def("get_size", &trjl::get_size, pydocs::DOC_TrajectoryList_get_size)
            .def("get_trajectory", &trjl::get_trajectory, py::return_value_policy::reference_internal,
                 pydocs::DOC_TrajectoryList_get_trajectory)
            .def("set_trajectory", &trjl::set_trajectory, pydocs::DOC_TrajectoryList_set_trajectory)
            .def("get_list", &trjl::get_list, pydocs::DOC_TrajectoryList_get_list)
            .def("get_batch", &trjl::get_batch, pydocs::DOC_TrajectoryList_get_batch)
            .def("sort_by_likelihood", &trjl::sort_by_likelihood,
                 pydocs::DOC_TrajectoryList_sort_by_likelihood)
            .def("sort_by_obs_count", &trjl::sort_by_obs_count, pydocs::DOC_TrajectoryList_sort_by_obs_count)
            .def("move_to_cpu", &trjl::move_to_cpu, pydocs::DOC_TrajectoryList_move_to_cpu)
            .def("move_to_gpu", &trjl::move_to_gpu, pydocs::DOC_TrajectoryList_move_to_gpu);
}
#endif

} /* namespace search */
