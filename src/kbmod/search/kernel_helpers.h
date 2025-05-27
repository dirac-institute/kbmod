/* Helper functions for interfacing with the .cu functions.
 * These functions are broken out into a separate file to centralize the
 * conditional logic for compiling and linking the CUDA code.
 */

#ifndef KERNEL_HELPERS_H_
#define KERNEL_HELPERS_H_

#include <string>


namespace search {

// Check that the package was built with CUDA support and there is a GPU available.
inline bool has_gpu();

// GPU Stat functions. The produces reasonable defaults when CUDA is not enabled.
void print_cuda_stats();
size_t get_gpu_total_memory();
size_t get_gpu_free_memory();
std::string stat_gpu_memory_mb();
bool validate_gpu(size_t req_memory = 0);

} /* namespace search */

#endif /* KERNEL_HELPERS_H_ */
