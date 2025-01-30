/*
 * kernel_memory.cu
 *
 * Helper functions for transfering KBMOD data to/from GPU.
 */

#ifndef KERNELS_MEMORY_CU_
#define KERNELS_MEMORY_CU_

#include <iostream>
#include "kernel_memory.h"

namespace search {

// Helpful debugging stats for when something crashes in the GPU.
void cuda_print_stats() {
    std::cout << "\n----- CUDA Debugging Log -----\n";

    int device_num, device_count;
    cudaGetDevice(&device_num);
    cudaGetDeviceCount(&device_count);
    std::cout << "Device: " << device_num << " [" << device_count << " devices available]\n";

    // Output information about the current memory usage.
    size_t free_mem, total_mem;
    unsigned int res = static_cast<unsigned int>(cudaMemGetInfo(&free_mem, &total_mem));
    if (res == 0) {
        double total_mb = ((double)total_mem) / 1048576.0;
        double free_mb = ((double)free_mem) / 1048576.0;
        std::cout << "Total Memory: " << total_mb << " MB\n";
        std::cout << "Used Memory: " << (total_mb - free_mb) << " MB\n";
        std::cout << "Free Memory: " << free_mb << " MB\n";
    } else {
        std::cout << "ERROR: Memory stats failed. Error code = " << res << "\n";
    }
}

// Check that we have a working GPU with enough memory.
bool cuda_check_gpu(size_t req_memory) {
    // Check that we can access the GPU itself.
    int device_num;
    unsigned int res = static_cast<unsigned int>(cudaGetDevice(&device_num));
    if (res != 0) {
        std::cout << "Unable to find GPU device.\n";
        return false;
    }

    // Check that we have enough memory.
    size_t free_mem, total_mem;
    res = static_cast<unsigned int>(cudaMemGetInfo(&free_mem, &total_mem));
    if (res != 0) {
        std::cout << "Unable to query GPU available memory.\n";
        return false;
    }
    if (free_mem < req_memory) {
        double free_mb = ((double)free_mem) / 1048576.0;
        std::cout << "Insufficient GPU memory free: " << free_mb << " MB\n";
        return false;
    }

    return true;
}

// ---------------------------------------
// --- Basic Memory Functions ------------
// ---------------------------------------

extern "C" void *allocate_gpu_block(uint64_t memory_size) {
    void *gpu_ptr;
    unsigned int res = static_cast<unsigned int>(cudaMalloc((void **)&gpu_ptr, memory_size));
    if ((res != 0) || (gpu_ptr == nullptr)) {
        cuda_print_stats();
        throw std::runtime_error("Unable to allocate GPU memory (" + std::to_string(memory_size) +
                                 " bytes). Error code = " + std::to_string(res));
    }
    return gpu_ptr;
}

extern "C" void free_gpu_block(void *gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    unsigned int res = static_cast<unsigned int>(cudaFree(gpu_ptr));
    if (res != 0) {
        cuda_print_stats();
        throw std::runtime_error("Unable to free GPU memory. Error code = " + std::to_string(res));
    }
}

extern "C" void copy_block_to_gpu(void *cpu_ptr, void *gpu_ptr, uint64_t memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");
    unsigned int res =
            static_cast<unsigned int>(cudaMemcpy(gpu_ptr, cpu_ptr, memory_size, cudaMemcpyHostToDevice));
    if (res != 0) {
        cuda_print_stats();
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
        cuda_print_stats();
        throw std::runtime_error("Unable to copy data to CPU (" + std::to_string(memory_size) +
                                 " bytes). Error code = " + std::to_string(res));
    }
}

} /* namespace search */

#endif /* KERNELS_MEMORY_CU_ */
