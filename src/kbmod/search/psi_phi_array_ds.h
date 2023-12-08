/*
 * psi_phi_array_ds.h
 *
 * The data structure for the interleaved psi/phi array.  The the data
 * structure and core functions are included in the header (and separated out
 * from the rest of the utility functions) to allow the CUDA files to import
 * only what they need.
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

namespace search {

/* PsiPhi is a simple struct to hold a named pair of psi and phi values. */
struct PsiPhi {
    float psi = 0.0;
    float phi = 0.0;
};

// Helper utility functions.
inline float encode_uint_scalar(float value, float min_val, float max_val, float scale) {
    return (value == NO_DATA) ? 0 : (std::max(std::min(value, max_val), min_val) - min_val) / scale + 1.0;
}

inline float decode_uint_scalar(float value, float min_val, float scale) {
    return (value == 0.0) ? NO_DATA : (value - 1.0) * scale + min_val;
}

/* PsiPhiArray is a class to hold the psi and phi arrays for the GPU
   as well as functions to do encoding and decoding.
   All functions used by CUDA are inlined and defined in the header.
*/
class PsiPhiArray {
public:
    explicit PsiPhiArray();
    explicit PsiPhiArray(int encode_bytes);
    virtual ~PsiPhiArray();

    void clear();

    // --- Getter functions (for Python interface) ----------------
    inline int get_num_bytes() { return num_bytes; }
    inline int get_num_times() { return num_times; }
    inline int get_width() { return width; }
    inline int get_height() { return height; }
    inline int get_pixels_per_image() { return pixels_per_image; }
    inline int get_num_entries() { return num_entries; }
    inline int get_total_array_size() { return total_array_size; }
    inline int get_block_size() { return block_size; }

    inline float get_psi_min_val() { return psi_min_val; }
    inline float get_psi_max_val() { return psi_max_val; }
    inline float get_psi_scale() { return psi_scale; }
    inline float get_phi_min_val() { return phi_min_val; }
    inline float get_phi_max_val() { return phi_max_val; }
    inline float get_phi_scale() { return phi_scale; }

    // Primary getter function for interaction (read the data).
    inline PsiPhi read_encoded_psi_phi(int time, int row, int col, bool from_gpu);

    // Setters for the utility functions to allocate the data.
    void set_meta_data(int new_num_times, int new_width, int new_height);
    void set_psi_scaling(float min_val, float max_val, float scale_val);
    void set_phi_scaling(float min_val, float max_val, float scale_val);

    // Array pointer functions needed for CUDA memory management. Should ONLY be called
    // by those utility functions. All other access should be done through the getters above.
    inline void* get_cpu_array_ptr() { return cpu_array_ptr; }
    inline void* get_gpu_array_ptr() { return gpu_array_ptr; }
    inline void set_cpu_array_ptr(void* new_ptr) { cpu_array_ptr = new_ptr; }
    inline void set_gpu_array_ptr(void* new_ptr) { gpu_array_ptr = new_ptr; }

private:
    inline int psi_index(int time, int row, int col) const {
        return 2 * (pixels_per_image * time + row * width + col);
    }
    
    inline int phi_index(int time, int row, int col) const {
        return 2 * (pixels_per_image * time + row * width + col) + 1;
    }

    // --- Attributes -----------------------------
    int num_times = 0;
    int width = 0;
    int height = 0;
    int pixels_per_image = 0;
    int num_entries = 0;
    int block_size = 0;
    long unsigned total_array_size = 0;

    // Pointers the CPU and GPU array.
    void* cpu_array_ptr = nullptr;
    void* gpu_array_ptr = nullptr;

    // Compression and scaling parameters of on GPU array.
    int num_bytes = -1;  // -1 (float), 1 (unit8) or 2 (unit16)

    float psi_min_val = FLT_MAX;
    float psi_max_val = -FLT_MAX;
    float psi_scale = 1.0;

    float phi_min_val = FLT_MAX;
    float phi_max_val = -FLT_MAX;
    float phi_scale = 1.0;
};

// read_encoded_psi_phi() is implemented in the header file so the CUDA files do not need
// to link against psi_phi_array.cpp.
inline PsiPhi PsiPhiArray::read_encoded_psi_phi(int time, int row, int col, bool from_gpu) {
    void* array_ptr = (from_gpu) ? gpu_array_ptr : cpu_array_ptr;
    assertm(array_ptr != nullptr, "Image data not allocated.");

    // Compute the in list index from the row, column, and time.
    int start_index = psi_index(time, row, col);
    assertm((start_index >= 0) && (start_index < num_entries), "Invalid index.");

    // Short circuit the typical case of float encoding.
    // No scaling or shifting done.
    PsiPhi result;
    if (num_bytes == -1) {
        result.psi = reinterpret_cast<float *>(array_ptr)[start_index];
        result.phi = reinterpret_cast<float *>(array_ptr)[start_index + 1];
    } else {
        // Handle the compressed encodings.
        float psi_value = (num_bytes == 1) ? (float)reinterpret_cast<uint8_t *>(array_ptr)[start_index]
                                           : (float)reinterpret_cast<uint16_t *>(array_ptr)[start_index];
        result.psi = decode_uint_scalar(psi_value, psi_min_val, psi_scale);

        float phi_value = (num_bytes == 1) ? (float)reinterpret_cast<uint8_t *>(array_ptr)[start_index + 1]
                                           : (float)reinterpret_cast<uint16_t *>(array_ptr)[start_index + 1];
        result.phi = decode_uint_scalar(phi_value, phi_min_val, phi_scale);
    }
    return result;
}

} /* namespace search */

#endif /* PSI_PHI_ARRAY_DS_ */
