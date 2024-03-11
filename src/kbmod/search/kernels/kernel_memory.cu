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

#include "../trajectory_list.h"

namespace search {


// ---------------------------------------
// --- Basic Memory Functions ------------
// ---------------------------------------

extern "C" void* allocate_gpu_block(unsigned long memory_size) {
    void *gpu_ptr;
    checkCudaErrors(cudaMalloc((void **)&gpu_ptr, memory_size));
    if (gpu_ptr == nullptr) throw std::runtime_error("Unable to allocate GPU memory.");
    return gpu_ptr;
}

extern "C" void free_gpu_block(void* gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    checkCudaErrors(cudaFree(gpu_ptr));
}

extern "C" void copy_block_to_gpu(void* cpu_ptr, void* gpu_ptr, unsigned long memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");

    checkCudaErrors(cudaMemcpy(gpu_ptr, cpu_ptr, memory_size, cudaMemcpyHostToDevice));
}

extern "C" void copy_block_to_cpu(void* cpu_ptr, void* gpu_ptr, unsigned long memory_size) {
    if (cpu_ptr == nullptr) throw std::runtime_error("Invalid CPU pointer");
    if (gpu_ptr == nullptr) throw std::runtime_error("Invalid GPU pointer");

    checkCudaErrors(cudaMemcpy(cpu_ptr, gpu_ptr, memory_size, cudaMemcpyDeviceToHost));
}

// ---------------------------------------
// --- Memory Functions ------------------
// ---------------------------------------


extern "C" float *move_floats_to_gpu(std::vector<float> &data) {
    unsigned long memory_size = data.size() * sizeof(float);

    float *gpu_ptr;
    checkCudaErrors(cudaMalloc((void **)&gpu_ptr, memory_size));
    checkCudaErrors(cudaMemcpy(gpu_ptr, data.data(), memory_size, cudaMemcpyHostToDevice));

    return gpu_ptr;
}

extern "C" void free_gpu_float_array(float *gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    checkCudaErrors(cudaFree(gpu_ptr));
}

extern "C" void *move_void_array_to_gpu(void *data_array, long unsigned memory_size) {
    if (data_array == nullptr) throw std::runtime_error("No data given.");
    if (memory_size == 0) throw std::runtime_error("Invalid size.");

    void *gpu_ptr;
    checkCudaErrors(cudaMalloc((void **)&gpu_ptr, memory_size));
    checkCudaErrors(cudaMemcpy(gpu_ptr, data_array, memory_size, cudaMemcpyHostToDevice));

    return gpu_ptr;
}

extern "C" void free_gpu_void_array(void *gpu_ptr) {
    if (gpu_ptr == nullptr) throw std::runtime_error("Trying to free nullptr.");
    checkCudaErrors(cudaFree(gpu_ptr));
}

extern "C" Trajectory *allocate_gpu_trajectory_list(long unsigned num_trj) {
    Trajectory *gpu_ptr;
    checkCudaErrors(cudaMalloc((void **)&gpu_ptr, num_trj * sizeof(Trajectory)));
    return gpu_ptr;
}

extern "C" void free_gpu_trajectory_list(Trajectory *gpu_ptr) { checkCudaErrors(cudaFree(gpu_ptr)); }

extern "C" void copy_trajectory_list(Trajectory *cpu_ptr, Trajectory *gpu_ptr, long unsigned num_trj,
                                     bool to_gpu) {
    if ((cpu_ptr == nullptr) || (gpu_ptr == nullptr)) throw std::runtime_error("Invalid pointer.");
    long unsigned memory_size = num_trj * sizeof(Trajectory);

    if (to_gpu) {
        checkCudaErrors(cudaMemcpy(gpu_ptr, cpu_ptr, memory_size, cudaMemcpyHostToDevice));
    } else {
        checkCudaErrors(cudaMemcpy(cpu_ptr, gpu_ptr, memory_size, cudaMemcpyDeviceToHost));
    }
}

} /* namespace search */

#endif /* KERNELS_MEMORY_CU_ */
