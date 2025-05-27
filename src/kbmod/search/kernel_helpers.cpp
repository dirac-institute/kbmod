/* Helper functions for interfacing with the .cu functions.
 * These functions are broken out into a separate file to centralize the
 * conditional logic for compiling and linking the CUDA code.
 */

#include <vector>

#include "kernel_helpers.h"
#include "pydocs/kernel_helper_docs.h"

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
#include "kernels/kernel_memory.h"
#endif

namespace search {
#ifdef HAVE_CUDA
/* The kernels.cu functions. */
extern "C" void SigmaGFilteredIndicesCU(float *values, int num_values, float sgl0, float sgl1, float sg_coeff,
                                        float width, int *idx_array, int *min_keep_idx, int *max_keep_idx);
#endif

inline bool has_gpu() {
#ifdef HAVE_CUDA
    return cude_device_count() > 0;
#else
    return false;
#endif
}

// --------------------------
// --- GPU Stat functions ---
// --------------------------

void print_cuda_stats() {
#ifdef HAVE_CUDA
    cuda_print_stats();
#else
    std::cout << "\n----- CUDA Debugging Log -----\n";
    std::cout << "CUDA not enabled.\n";
#endif
}

size_t get_gpu_total_memory() {
#ifdef HAVE_CUDA
    return gpu_total_memory();
#else
    // package was built without a GPU.
    return 0;
#endif
}

size_t get_gpu_free_memory() {
#ifdef HAVE_CUDA
    return gpu_free_memory();
#else
    // package was built without a GPU.
    return 0;
#endif
}

std::string stat_gpu_memory_mb() {
    double total_mb = (double)get_gpu_total_memory() / 1048576.0;
    double free_mb = (double)get_gpu_free_memory() / 1048576.0;
    return ("GPU: " + std::to_string(free_mb) + " MB free of " + std::to_string(total_mb) + " MB total.");
}

bool validate_gpu(size_t req_memory) {
#ifdef HAVE_CUDA
    return cuda_check_gpu(req_memory);
#else
    // package was built without a GPU.
    return false;
#endif
}

/* Used for testing SigmaGFilteredIndicesCU for python
 *
 * Return the list of indices from the values array such that those elements
 * pass the sigmaG filtering defined by percentiles [sgl0, sgl1] with coefficient
 * sigma_g_coeff and a multiplicative factor of width.
 *
 * The vector values is passed by value to create a local copy which will be modified by
 * SigmaGFilteredIndicesCU.
 */
std::vector<int> sigmaGFilteredIndices(std::vector<float> values, float sgl0, float sgl1, float sigma_g_coeff,
                                       float width) {
    int num_values = values.size();
    std::vector<int> idx_array(num_values, 0);
    int min_keep_idx = 0;
    int max_keep_idx = num_values - 1;

#ifdef HAVE_CUDA
    SigmaGFilteredIndicesCU(values.data(), num_values, sgl0, sgl1, sigma_g_coeff, width, idx_array.data(),
                            &min_keep_idx, &max_keep_idx);
#else
    throw std::runtime_error("Non-GPU SigmaGFilteredIndicesCU is not implemented.");
#endif

    // Copy the result into a vector and return it.
    std::vector<int> result;
    for (int i = min_keep_idx; i <= max_keep_idx; ++i) {
        result.push_back(idx_array[i]);
    }
    return result;
}

#ifdef Py_PYTHON_H
static void kernel_helper_bindings(py::module &m) {
    m.def("kb_has_gpu", &has_gpu, "Check if GPU is available");
    m.def("sigmag_filtered_indices", &search::sigmaGFilteredIndices);
    m.def("print_cuda_stats", &search::print_cuda_stats, pydocs::DOC_print_cuda_stats);
    m.def("get_gpu_total_memory", &search::get_gpu_total_memory, pydocs::DOC_get_gpu_total_memory);
    m.def("get_gpu_free_memory", &search::get_gpu_free_memory, pydocs::DOC_get_gpu_free_memory);
    m.def("stat_gpu_memory_mb", &search::stat_gpu_memory_mb, pydocs::DOC_stat_gpu_memory_mb);
    m.def("validate_gpu", &search::validate_gpu, py::arg("req_memory") = 0, pydocs::DOC_validate_gpu);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
