/*
 * gpu_array.h
 *
 * A templated fixed-size GPU array that provides dynamic movement of data
 * to and from the GPU with safety checks. This serves as a wrapper that can
 * be called from non-CUDA code (though some functions will fail if the
 * CUDA code is not enabled).
 *
 * Some attributes:
 *   - Lazy memory allocation on the GPU.
 *   - Automatic freeing of GPU memory when object is destroyed.
 *   - Only performs shallow copies of the data when moving to/from GPU.
 */

#ifndef GPU_ARRAY_H_
#define GPU_ARRAY_H_

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
#include "kernels/kernel_memory.h"
#endif

namespace search {

template <typename T>
class GPUArray {
public:
    // Use lazy allocation for uninitialized arrays.
    explicit GPUArray(uint64_t num_items, bool allocate_now = false) : gpu_ptr(nullptr) {
        size = num_items;
        memory_size = size * sizeof(T);

        if (allocate_now) allocate_gpu_memory();
    }

    // Default constructor is an empty array.
    explicit GPUArray() : GPUArray(0, false) {}

    // Copy the vector's data to the GPU directly.
    explicit GPUArray(std::vector<T>& data) : gpu_ptr(nullptr) {
        size = data.size();
        memory_size = size * sizeof(T);
        copy_vector_to_gpu(data);
    }

    // Do not allow copying or assignment. This prevents the a copy of the array
    // being used after another copy has gone out of scope and freed the GPU memory.
    GPUArray(GPUArray&) = delete;
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;

    virtual ~GPUArray() {
        if (gpu_ptr != nullptr) free_gpu_memory();
    }

    // --- Basic Getters --------------------
    inline bool on_gpu() const { return gpu_ptr != nullptr; }
    inline uint64_t get_size() const { return size; }
    inline uint64_t get_memory_size() const { return memory_size; }
    inline T* get_ptr() { return gpu_ptr; }

    // Resizing an array with allocated GPU memory must use the destructive flag
    // in which case it frees the memory.
    void resize(uint64_t new_size, bool destructive = false) {
        if (new_size == size) return;  // Nothing to do.
        if (gpu_ptr != nullptr) {
            if (!destructive) throw std::runtime_error("Unable to resize array on GPU");
            free_gpu_memory();
        }

        size = new_size;
        memory_size = new_size * sizeof(T);
    }

    // --- Basic memory functions -----------
    void allocate_gpu_memory() {
        if (gpu_ptr != nullptr) throw std::runtime_error("GPU data already allocated");
#ifdef HAVE_CUDA
        gpu_ptr = (T*)allocate_gpu_block(memory_size);
#endif
        if (gpu_ptr == nullptr) throw std::runtime_error("Unable to allocate GPU memory");
    }

    void free_gpu_memory() {
        if (gpu_ptr == nullptr) return;  // Nothing to do.
#ifdef HAVE_CUDA
        free_gpu_block((void*)gpu_ptr);
#else
        throw std::runtime_error("GPU needed to free GPU memory");
#endif
        gpu_ptr = nullptr;
    }

    // Copy an entire vector into the corresponding GPU memory.
    void copy_vector_to_gpu(std::vector<T>& data) {
        if (data.size() != size) throw std::runtime_error("Vector size mismatch");
        if (gpu_ptr == nullptr) allocate_gpu_memory();
#ifdef HAVE_CUDA
        copy_block_to_gpu((void*)data.data(), (void*)gpu_ptr, memory_size);
#endif
    }

    // Copy the full GPU memory into a CPU vector.
    void copy_gpu_to_vector(std::vector<T>& data) {
        if (data.size() != size) throw std::runtime_error("Vector size mismatch");
        if (gpu_ptr == nullptr) throw std::runtime_error("No GPU data allocated");
#ifdef HAVE_CUDA
        copy_block_to_cpu((void*)data.data(), (void*)gpu_ptr, memory_size);
#endif
    }

    // Copy an array into a subset of the full GPU memory.
    void copy_array_into_subset_of_gpu(T* data_ptr, uint64_t start_index, uint64_t data_size) {
        if (data_size == 0) return;  // Nothing to do.
        if (data_ptr == nullptr) throw std::runtime_error("Copying from nullptr.");
        if (data_size + start_index > size) {
            throw std::runtime_error("Data copied to GPU exceeds length of memory.");
        }

#ifdef HAVE_CUDA
        if (gpu_ptr == nullptr) allocate_gpu_memory();
        T* start_ptr = gpu_ptr + start_index;
        uint64_t data_memory = data_size * sizeof(T);
        copy_block_to_gpu((void*)data_ptr, (void*)start_ptr, data_memory);
#endif
    }

    // Copy a vector into a subset of the full GPU memory.
    // This is used when loading data into the GPU in pieces.
    void copy_vector_into_subset_of_gpu(std::vector<T>& data, uint64_t start_index) {
        copy_array_into_subset_of_gpu(data.data(), start_index, data.size());
    }

private:
    uint64_t size;
    uint64_t memory_size;
    T* gpu_ptr;
};

} /* namespace search */

#endif /* GPU_ARRAY_H_ */
