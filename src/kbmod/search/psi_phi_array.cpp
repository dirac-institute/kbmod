#include "psi_phi_array_ds.h"
#include "psi_phi_array_utils.h"
#include "pydocs/psi_phi_array_docs.h"

namespace search {

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
extern "C" void device_allocate_psi_phi_array(PsiPhiArray* data);

extern "C" void device_free_psi_phi_array(PsiPhiArray* data);
#endif

// -------------------------------------------------------
// --- Implementation of core data structure functions ---
// -------------------------------------------------------

PsiPhiArray::PsiPhiArray() {}

PsiPhiArray::~PsiPhiArray() {
    if (cpu_array_ptr != nullptr) {
        free(cpu_array_ptr);
    }
#ifdef HAVE_CUDA
    if (gpu_array_ptr != nullptr) {
        device_free_psi_phi_array(this);
    }
#endif    
}

void PsiPhiArray::clear() {
    // Free all used memory on CPU and GPU.
    if (cpu_array_ptr != nullptr) {
        free(cpu_array_ptr);
        cpu_array_ptr = nullptr;
    }
#ifdef HAVE_CUDA
    if (gpu_array_ptr != nullptr) {
        device_free_psi_phi_array(this);
        gpu_array_ptr = nullptr;
    }
#endif

    // Reset the meta data except the encoding information.
    meta_data.num_times = 0;
    meta_data.width = 0;
    meta_data.height = 0;
    meta_data.pixels_per_image = 0;
    meta_data.num_entries = 0;
    meta_data.total_array_size = 0;

    meta_data.psi_min_val = FLT_MAX;
    meta_data.psi_max_val = -FLT_MAX;
    meta_data.psi_scale = 1.0;

    meta_data.phi_min_val = FLT_MAX;
    meta_data.phi_max_val = -FLT_MAX;
    meta_data.phi_scale = 1.0;
}

void PsiPhiArray::set_meta_data(int new_num_bytes, int new_num_times, int new_height, int new_width) {
    // Validity checking of parameters.
    if (new_num_bytes != -1 && new_num_bytes != 1 && new_num_bytes != 2 && new_num_bytes != 4) {
        throw std::runtime_error("Invalid setting of num_bytes. Must be (-1, 1, 2, or 4).");
    }
    if (new_num_times <= 0) throw std::runtime_error("Invalid num_times passed to set_meta_data.\n");
    if (new_width <= 0) throw std::runtime_error("Invalid width passed to set_meta_data.\n");
    if (new_height <= 0) throw std::runtime_error("Invalid height passed to set_meta_data.\n");

    // Check that we do not have an array allocated.
    if (cpu_array_ptr != nullptr) {
        throw std::runtime_error("Cannot change meta data with allocated arrays. Call clear() first.");
    }

    meta_data.num_bytes = new_num_bytes;
    if (meta_data.num_bytes == 1) {
        meta_data.block_size = sizeof(uint8_t);
    } else if (meta_data.num_bytes == 2) {
        meta_data.block_size = sizeof(uint16_t);
    } else {
        meta_data.num_bytes = -1;
        meta_data.block_size = sizeof(float);
    }

    meta_data.num_times = new_num_times;
    meta_data.width = new_width;
    meta_data.height = new_height;
    meta_data.pixels_per_image = meta_data.width * meta_data.height;
    meta_data.num_entries = 2 * meta_data.pixels_per_image * meta_data.num_times;
    meta_data.total_array_size = meta_data.block_size * meta_data.num_entries;
}

void PsiPhiArray::set_psi_scaling(float min_val, float max_val, float scale_val) {
    if (min_val > max_val) throw std::runtime_error("Min value needs to be < max value");
    if (scale_val <= 0) throw std::runtime_error("Scale value must be greater than zero.");
    meta_data.psi_min_val = min_val;
    meta_data.psi_max_val = max_val;
    meta_data.psi_scale = scale_val;
}

void PsiPhiArray::set_phi_scaling(float min_val, float max_val, float scale_val) {
    if (min_val > max_val) throw std::runtime_error("Min value needs to be < max value");
    if (scale_val <= 0) throw std::runtime_error("Scale value must be greater than zero.");
    meta_data.phi_min_val = min_val;
    meta_data.phi_max_val = max_val;
    meta_data.phi_scale = scale_val;
}

PsiPhi PsiPhiArray::read_psi_phi(int time, int row, int col) {
    PsiPhi result = {NO_DATA, NO_DATA};

    // Array allocation and bounds checking.
    if ((cpu_array_ptr == nullptr) || (row < 0) || (col < 0) || (row >= meta_data.height) ||
        (col >= meta_data.width)) {
        return result;
    }

    // Compute the in-list index from the row, column, and time.
    int start_index = 2 * (meta_data.pixels_per_image * time + row * meta_data.width + col);

    if (meta_data.num_bytes == -1) {
        // Short circuit the typical case of float encoding.
        // No scaling or shifting done.
        result.psi = reinterpret_cast<float*>(cpu_array_ptr)[start_index];
        result.phi = reinterpret_cast<float*>(cpu_array_ptr)[start_index + 1];
    } else {
        // Handle the compressed encodings.
        float psi_value = (meta_data.num_bytes == 1)
                                  ? (float)reinterpret_cast<uint8_t*>(cpu_array_ptr)[start_index]
                                  : (float)reinterpret_cast<uint16_t*>(cpu_array_ptr)[start_index];
        result.psi = (psi_value == 0.0) ? NO_DATA
                                        : (psi_value - 1.0) * meta_data.psi_scale + meta_data.psi_min_val;

        float phi_value = (meta_data.num_bytes == 1)
                                  ? (float)reinterpret_cast<uint8_t*>(cpu_array_ptr)[start_index + 1]
                                  : (float)reinterpret_cast<uint16_t*>(cpu_array_ptr)[start_index + 1];
        result.phi = (phi_value == 0.0) ? NO_DATA
                                        : (phi_value - 1.0) * meta_data.phi_scale + meta_data.phi_min_val;
    }
    return result;
}

// -------------------------------------------
// --- Implementation of utility functions ---
// -------------------------------------------

// Compute the min, max, and scale parameter from the a vector of image data.
std::array<float, 3> compute_scale_params_from_image_vect(const std::vector<RawImage>& imgs, int num_bytes) {
    int num_images = imgs.size();

    // Do a linear pass through the data to compute the scaling parameters for psi and phi.
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (int i = 0; i < num_images; ++i) {
        std::array<float, 2> bnds = imgs[i].compute_bounds();
        if (bnds[0] < min_val) min_val = bnds[0];
        if (bnds[1] > max_val) max_val = bnds[1];
    }

    // Set the scale if we are encoding the values.
    float scale = 1.0;
    if (num_bytes == 1 || num_bytes == 2) {
        float width = (max_val - min_val);
        if (width < 1e-6) width = 1e-6;  // Avoid a zero width.

        long int num_values = (1 << (8 * num_bytes)) - 1;
        scale = width / (double)num_values;
    }

    return {min_val, max_val, scale};
}

template <typename T>
void set_encode_cpu_psi_phi_array(PsiPhiArray& data, const std::vector<RawImage>& psi_imgs,
                                  const std::vector<RawImage>& phi_imgs) {
    if (data.get_cpu_array_ptr() != nullptr) {
        throw std::runtime_error("CPU PsiPhi already allocated.");
    }
    T* encoded = (T*)malloc(data.get_total_array_size());

    // Create a safe maximum that is slightly less than the true max to avoid
    // rollover of the unsigned integer.
    float safe_max_psi = data.get_psi_max_val() - data.get_psi_scale() / 100.0;
    float safe_max_phi = data.get_phi_max_val() - data.get_phi_scale() / 100.0;

    int current_index = 0;
    int num_bytes = data.get_num_bytes();
    for (int t = 0; t < data.get_num_times(); ++t) {
        for (int row = 0; row < data.get_height(); ++row) {
            for (int col = 0; col < data.get_width(); ++col) {
                float psi_value = psi_imgs[t].get_pixel({row, col});
                float phi_value = phi_imgs[t].get_pixel({row, col});

                // Handle the encoding for the different values.
                if (num_bytes == 1 || num_bytes == 2) {
                    psi_value = encode_uint_scalar(psi_value, data.get_psi_min_val(), safe_max_psi,
                                                   data.get_psi_scale());
                    phi_value = encode_uint_scalar(phi_value, data.get_phi_min_val(), safe_max_phi,
                                                   data.get_phi_scale());
                }

                encoded[current_index++] = static_cast<T>(psi_value);
                encoded[current_index++] = static_cast<T>(phi_value);
            }
        }
    }

    data.set_cpu_array_ptr((void*)encoded);
}

void set_float_cpu_psi_phi_array(PsiPhiArray& data, const std::vector<RawImage>& psi_imgs,
                                 const std::vector<RawImage>& phi_imgs) {
    if (data.get_cpu_array_ptr() != nullptr) {
        throw std::runtime_error("CPU PsiPhi already allocated.");
    }
    float* encoded = (float*)malloc(data.get_total_array_size());

    int current_index = 0;
    for (int t = 0; t < data.get_num_times(); ++t) {
        for (int row = 0; row < data.get_height(); ++row) {
            for (int col = 0; col < data.get_width(); ++col) {
                encoded[current_index++] = psi_imgs[t].get_pixel({row, col});
                encoded[current_index++] = phi_imgs[t].get_pixel({row, col});
            }
        }
    }

    data.set_cpu_array_ptr((void*)encoded);
}

void fill_psi_phi_array(PsiPhiArray& result_data, int num_bytes, const std::vector<RawImage>& psi_imgs,
                        const std::vector<RawImage>& phi_imgs) {
    if (result_data.get_cpu_array_ptr() != nullptr) {
        return;
    }

    // Set the meta data and do a bunch of validity checks.
    int num_times = psi_imgs.size();
    if (num_times <= 0) throw std::runtime_error("Trying to fill PsiPhi from empty vectors.");
    if (num_times != phi_imgs.size()) throw std::runtime_error("Size mismatch between psi and phi.");

    int width = phi_imgs[0].get_width();
    int height = phi_imgs[0].get_height();
    result_data.set_meta_data(num_bytes, num_times, height, width);

    if (result_data.get_num_bytes() == 1 || result_data.get_num_bytes() == 2) {
        // Compute the scaling parameters needed for encoding.
        std::array<float, 3> psi_params =
                compute_scale_params_from_image_vect(psi_imgs, result_data.get_num_bytes());
        result_data.set_psi_scaling(psi_params[0], psi_params[1], psi_params[2]);

        std::array<float, 3> phi_params =
                compute_scale_params_from_image_vect(phi_imgs, result_data.get_num_bytes());
        result_data.set_phi_scaling(phi_params[0], phi_params[1], phi_params[2]);

        // Do the local encoding.
        if (result_data.get_num_bytes() == 1) {
            set_encode_cpu_psi_phi_array<uint8_t>(result_data, psi_imgs, phi_imgs);
        } else {
            set_encode_cpu_psi_phi_array<uint16_t>(result_data, psi_imgs, phi_imgs);
        }
    } else {
        // Just interleave psi and phi images.
        set_float_cpu_psi_phi_array(result_data, psi_imgs, phi_imgs);
    }

#ifdef HAVE_CUDA
    // Create a copy of the encoded data in GPU memory.
    device_allocate_psi_phi_array(&result_data);
#endif
}

// -------------------------------------------
// --- Python definitions --------------------
// -------------------------------------------

#ifdef Py_PYTHON_H
static void psi_phi_array_binding(py::module& m) {
    using ppa = search::PsiPhiArray;

    py::class_<search::PsiPhi>(m, "PsiPhi", pydocs::DOC_PsiPhi))
            .def(py::init<>())
            .def_readwrite("psi", &search::PsiPhi::psi)
            .def_readwrite("phi", &search::PsiPhi::phi);

    py::class_<ppa>(m, "PsiPhiArray", DOC_PsiPhiArray)
            .def(py::init<>())
            .def_property_readonly("num_bytes", &ppa::get_num_bytes, DOC_PsiPhiArray_get_num_bytes)
            .def_property_readonly("num_times", &ppa::get_num_times, DOC_PsiPhiArray_get_num_times)
            .def_property_readonly("width", &ppa::get_width, DOC_PsiPhiArray_get_width)
            .def_property_readonly("height", &ppa::get_height, DOC_PsiPhiArray_get_height)
            .def_property_readonly("pixels_per_image", &ppa::get_pixels_per_image, DOC_PsiPhiArray_get_pixels_per_image)
            .def_property_readonly("num_entries", &ppa::get_num_entries, DOC_PsiPhiArray_get_num_entries)
            .def_property_readonly("total_array_size", &ppa::get_total_array_size, DOC_PsiPhiArray_get_total_array_size)
            .def_property_readonly("block_size", &ppa::get_block_size, DOC_PsiPhiArray_get_block_size)
            .def_property_readonly("psi_min_val", &ppa::get_psi_min_val, DOC_PsiPhiArray_get_psi_min_val)
            .def_property_readonly("psi_max_val", &ppa::get_psi_max_val, DOC_PsiPhiArray_get_psi_max_val)
            .def_property_readonly("psi_scale", &ppa::get_psi_scale, DOC_PsiPhiArray_get_psi_scale)
            .def_property_readonly("phi_min_val", &ppa::get_phi_min_val, DOC_PsiPhiArray_get_phi_min_val)
            .def_property_readonly("phi_max_val", &ppa::get_phi_max_val, DOC_PsiPhiArray_get_phi_max_val)
            .def_property_readonly("phi_scale", &ppa::get_phi_scale, DOC_PsiPhiArray_get_phi_scale)
            .def_property_readonly("cpu_array_allocated", &ppa::cpu_array_allocated, DOC_PsiPhiArray_get_cpu_array_allocated)
            .def_property_readonly("gpu_array_allocated", &ppa::gpu_array_allocated, DOC_PsiPhiArray_get_gpu_array_allocated)
            .def("set_meta_data", &ppa::set_meta_data, DOC_PsiPhiArray_set_meta_data)
            .def("clear", &ppa::clear, DOC_PsiPhiArray_clear)
            .def("read_psi_phi", &ppa::read_psi_phi, DOC_PsiPhiArray_read_psi_phi);
    m.def("compute_scale_params_from_image_vect", &search::compute_scale_params_from_image_vect);
    m.def("decode_uint_scalar", &search::decode_uint_scalar);
    m.def("encode_uint_scalar", &search::encode_uint_scalar);
    m.def("fill_psi_phi_array", &search::fill_psi_phi_array, DOC_PsiPhiArray_fill_psi_phi_array);
}
#endif

} /* namespace search */
