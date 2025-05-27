/*
 * kernel_memory.h
 *
 * Helper functions for transfering KBMOD data to/from GPU. The functions throw
 * std::runtime_error that can be propagated back to python via pybind11.
 */

#ifndef KERNELS_MEMORY_H_
#define KERNELS_MEMORY_H_

#include <string.h>
#include <stdexcept>
#include <stdint.h>

namespace search {

int cuda_device_count();
void cuda_print_stats();

size_t gpu_total_memory();
size_t gpu_free_memory();

// Check that we have a working GPU with enough memory.
bool cuda_check_gpu(size_t req_memory);

// ---------------------------------------
// --- Basic Memory Functions ------------
// ---------------------------------------

extern "C" void* allocate_gpu_block(uint64_t memory_size);

extern "C" void free_gpu_block(void* gpu_ptr);

extern "C" void copy_block_to_gpu(void* cpu_ptr, void* gpu_ptr, uint64_t memory_size);

extern "C" void copy_block_to_cpu(void* cpu_ptr, void* gpu_ptr, uint64_t memory_size);

} /* namespace search */

#endif /* KERNELS_MEMORY_CU_ */
