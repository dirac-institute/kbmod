/*
 * kernel_memory.cu
 *
 * Helper functions for transfering KBMOD data to/from GPU.
 */

#ifndef KERNELS_MEMORY_CU_
#define KERNELS_MEMORY_CU_

#include "kernel_memory.h"

namespace search {

// ---------------------------------------
// --- Basic Memory Functions ------------
// ---------------------------------------

extern "C" void *allocate_gpu_block(uint64_t memory_size) {
    void *gpu_ptr;
    unsigned int res = static_cast<unsigned int>(cudaMalloc((void **)&gpu_ptr, memory_size));
    if ((res != 0) || (gpu_ptr == nullptr)) {
        throw std::runtime_error("Unable to allocate GPU memory (" + std::to_string(memory_size) +
                                 " bytes). Error code = " + std::to_string(res));
    }
    return gpu_ptr;
}

extern "C" void free_gpu_block(void *gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    unsigned int res = static_cast<unsigned int>(cudaFree(gpu_ptr));
    if (res != 0) {
        throw std::runtime_error("Unable to free GPU memory. Error code = " + std::to_string(res));
    }
}

extern "C" void copy_block_to_gpu(void *cpu_ptr, void *gpu_ptr, uint64_t memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");
    unsigned int res =
            static_cast<unsigned int>(cudaMemcpy(gpu_ptr, cpu_ptr, memory_size, cudaMemcpyHostToDevice));
    if (res != 0) {
        throw std::runtime_error("Unable to copy data to GPU (" + std::to_string(memory_size) +
                                 " bytes). Error code = " + std::to_string(res));
    }
}

extern "C" void copy_block_to_cpu(void *cpu_ptr, void *gpu_ptr, uint64_t memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");
    unsigned int res =
            static_cast<unsigned int>(cudaMemcpy(cpu_ptr, gpu_ptr, memory_size, cudaMemcpyDeviceToHost));
    if (res != 0) {
        throw std::runtime_error("Unable to copy data to CPU (" + std::to_string(memory_size) +
                                 " bytes). Error code = " + std::to_string(res));
    }
}

} /* namespace search */

#endif /* KERNELS_MEMORY_CU_ */
