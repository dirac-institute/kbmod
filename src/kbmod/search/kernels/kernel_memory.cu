/*
 * kernel_memory.cu
 *
 * Helper functions for transfering KBMOD data to/from GPU.
 */

#ifndef KERNELS_MEMORY_CU_
#define KERNELS_MEMORY_CU_

#include <stdexcept>
#include <float.h>
#include <vector>

#include "cuda_errors.h"

namespace search {

// ---------------------------------------
// --- Basic Memory Functions ------------
// ---------------------------------------

extern "C" void *allocate_gpu_block(unsigned long memory_size) {
    void *gpu_ptr;
    checkCudaErrors(cudaMalloc((void **)&gpu_ptr, memory_size));
    if (gpu_ptr == nullptr) throw std::runtime_error("Unable to allocate GPU memory.");
    return gpu_ptr;
}

extern "C" void free_gpu_block(void *gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    checkCudaErrors(cudaFree(gpu_ptr));
}

extern "C" void copy_block_to_gpu(void *cpu_ptr, void *gpu_ptr, unsigned long memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");

    checkCudaErrors(cudaMemcpy(gpu_ptr, cpu_ptr, memory_size, cudaMemcpyHostToDevice));
}

extern "C" void copy_block_to_cpu(void *cpu_ptr, void *gpu_ptr, unsigned long memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");

    checkCudaErrors(cudaMemcpy(cpu_ptr, gpu_ptr, memory_size, cudaMemcpyDeviceToHost));
}

} /* namespace search */

#endif /* KERNELS_MEMORY_CU_ */
