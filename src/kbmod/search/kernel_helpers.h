/* Helper functions for testing functions in the .cu files from Python. */
#ifndef KERNEL_HELPERS_H_
#define KERNEL_HELPERS_H_

#include <string>

#include "logging.h"

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
#include "kernels/kernel_memory.h"
#endif

namespace search {

void print_cuda_stats();

size_t get_gpu_total_memory();

size_t get_gpu_free_memory();

std::string stat_gpu_memory_mb();

bool validate_gpu(size_t req_memory = 0);

} /* namespace search */

#endif /* KERNEL_HELPERS_H_ */
