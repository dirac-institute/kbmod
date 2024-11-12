/*
 * psi_phi_array_ds.h
 *
 * The data structure for the raw data needed for the search algorith,
 * including the psi/phi values and the zeroed times. The the data
 * structure and core functions are included in the header (and separated out
 * from the rest of the utility functions) to allow the CUDA files to import
 * only what they need.
 *
 * The data structure allocates memory on both the CPU and GPU for the
 * arrays and maintains ownership of the pointers until clear() is called
 * the object's destructor is called. This allows the object to be passed
 * repeatedly to the on-device search without reallocating and copying the
 * memory on the GPU. All arrays are stored as pointers (instead of vectors)
 * for compatibility with CUDA.
 *
 * Created on: Dec 5, 2023
 */

#ifndef PSI_PHI_ARRAY_DS_
#define PSI_PHI_ARRAY_DS_

#include <cmath>
#include <stdio.h>
#include <float.h>
#include <vector>

#include "common.h"
#include "gpu_array.h"

namespace search {

/* PsiPhi is a simple struct to hold a named pair of psi and phi values. */
struct PsiPhi {
    float psi = 0.0;
    float phi = 0.0;
};

// The struct of meta data for the PsiPhiArray.
struct PsiPhiArrayMeta {
    uint64_t num_times = 0;
    uint64_t width = 0;
    uint64_t height = 0;
    uint64_t pixels_per_image = 0;
    uint64_t num_entries = 0;
    uint64_t block_size = 0;  // Actual memory used per entry.
    uint64_t total_array_size = 0;
};

/* PsiPhiArray is a class to hold the psi and phi arrays for the CPU and GPU as well as
   the meta data and functions to do encoding and decoding on CPU.
*/
class PsiPhiArray {
public:
    explicit PsiPhiArray();
    virtual ~PsiPhiArray();

    // Disallow copying and assignment to avoid invalid GPU memory pointers.
    PsiPhiArray(PsiPhiArray&) = delete;
    PsiPhiArray(const PsiPhiArray&) = delete;
    PsiPhiArray& operator=(PsiPhiArray&) = delete;
    PsiPhiArray& operator=(const PsiPhiArray&) = delete;

    void clear();

    inline PsiPhiArrayMeta& get_meta_data() { return meta_data; }

    // --- Getter functions (for Python interface) ----------------
    inline bool on_gpu() { return data_on_gpu; }
    inline uint64_t get_num_times() { return meta_data.num_times; }
    inline uint64_t get_width() { return meta_data.width; }
    inline uint64_t get_height() { return meta_data.height; }
    inline uint64_t get_pixels_per_image() { return meta_data.pixels_per_image; }
    inline uint64_t get_num_entries() { return meta_data.num_entries; }
    inline uint64_t get_total_array_size() { return meta_data.total_array_size; }
    inline uint64_t get_block_size() { return meta_data.block_size; }

    inline bool cpu_array_allocated() { return cpu_array_ptr != nullptr; }
    inline bool gpu_array_allocated() { return gpu_array_ptr != nullptr; }

    // Primary getter functions for interaction (read the data).
    PsiPhi read_psi_phi(uint64_t time_index, int row, int col);
    double read_time(uint64_t time_index);

    // Setters for the utility functions to allocate the data.
    void set_meta_data(uint64_t new_num_times, uint64_t new_height, uint64_t new_width);
    void set_time_array(const std::vector<double>& times);

    // Functions for loading / unloading data onto GPU.
    void move_to_gpu();
    void clear_from_gpu();

    // Should ONLY be called by the utility functions.
    inline void* get_cpu_array_ptr() { return cpu_array_ptr; }
    inline void* get_gpu_array_ptr() { return gpu_array_ptr; }
    inline void set_cpu_array_ptr(void* new_ptr) { cpu_array_ptr = new_ptr; }

    inline double* get_cpu_time_array_ptr() { return cpu_time_array.data(); }
    inline double* get_gpu_time_array_ptr() { return gpu_time_array.get_ptr(); }

private:
    PsiPhiArrayMeta meta_data;
    bool data_on_gpu;

    // Pointers to the arrays
    void* cpu_array_ptr = nullptr;
    void* gpu_array_ptr = nullptr;
    std::vector<double> cpu_time_array;
    GPUArray<double> gpu_time_array;
};

} /* namespace search */

#endif /* PSI_PHI_ARRAY_DS_ */
