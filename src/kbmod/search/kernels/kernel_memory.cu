/*
 * kernel_memory.cu
 *
 * Helper functions for transfering KBMOD data to/from GPU.
 */

#ifndef KERNELS_MEMORY_CU_
#define KERNELS_MEMORY_CU_

#include <float.h>
#include <string.h>
#include <stdexcept>
#include <vector>

#include "cuda_errors.h"

namespace search {

// ---------------------------------------
// --- Basic Memory Functions ------------
// ---------------------------------------

extern "C" void *allocate_gpu_block(unsigned long memory_size) {
    void *gpu_ptr;
    cudaError_t res = cudaMalloc((void **)&gpu_ptr, memory_size);
    if (res != 0) throw std::runtime_error("Unable to allocate GPU memory. Error = " + std::to_string(res));
    return gpu_ptr;
}

extern "C" void free_gpu_block(void *gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    cudaError_t res = cudaFree(gpu_ptr);
    if (res != 0) throw std::runtime_error("Unable to free GPU memory. Error = " + std::to_string(res));
}

extern "C" void copy_block_to_gpu(void *cpu_ptr, void *gpu_ptr, unsigned long memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");

    cudaError_t res = cudaMemcpy(gpu_ptr, cpu_ptr, memory_size, cudaMemcpyHostToDevice);
    if (res != 0) throw std::runtime_error("Unable to copy data to GPU. Error = " + std::to_string(res));
}

extern "C" void copy_block_to_cpu(void *cpu_ptr, void *gpu_ptr, unsigned long memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");

    cudaError_t res = cudaMemcpy(cpu_ptr, gpu_ptr, memory_size, cudaMemcpyDeviceToHost);
    if (res != 0) throw std::runtime_error("Unable to copy data to CPU. Error = " + std::to_string(res));
}

} /* namespace search */

#endif /* KERNELS_MEMORY_CU_ */
