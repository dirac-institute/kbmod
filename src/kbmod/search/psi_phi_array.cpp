#include "psi_phi_array_ds.h"
#include "psi_phi_array_utils.h"

namespace search {

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
extern "C" void device_allocate_psi_phi_array(PsiPhiArray* data);

extern "C" void device_free_psi_phi_array(PsiPhiArray* data);
#endif

// -------------------------------------------------------
// --- Implementation of core data structure functions ---
// -------------------------------------------------------

PsiPhiArray::PsiPhiArray() : num_bytes(-1) {
    block_size = sizeof(float);
}
    
PsiPhiArray::PsiPhiArray(int encode_bytes) : num_bytes(encode_bytes) {
    if (num_bytes == 1) {
        block_size = sizeof(uint8_t);
    } else if (num_bytes == 2) {
        block_size = sizeof(uint16_t);
    } else {
        num_bytes = -1;
        block_size = sizeof(float);
    }
}
    
PsiPhiArray::~PsiPhiArray() {
    clear();
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
    num_times = 0;
    width = 0;
    height = 0;
    pixels_per_image = 0;
    num_entries = 0;
    total_array_size = 0;

    psi_min_val = FLT_MAX;
    psi_max_val = -FLT_MAX;
    psi_scale = 1.0;

    phi_min_val = FLT_MAX;
    phi_max_val = -FLT_MAX;
    phi_scale = 1.0;
}

void PsiPhiArray::set_meta_data(int new_num_times, int new_width, int new_height) {
    assertm(num_times > 0 && width > 0 && height > 0,
            "Invalid metadata provided to PsiPhiArray");
    assertm(cpu_array_ptr == nullptr, "Cannot change meta data with allocated arrays. Call clear() first.");

    num_times = new_num_times;
    width = new_width;
    height = new_height;
    pixels_per_image = width * height;
    num_entries = 2 * pixels_per_image * num_times;
    total_array_size = block_size * num_entries;    
}

void PsiPhiArray::set_psi_scaling(float min_val, float max_val, float scale_val) {
    assertm(min_val < max_val, "Min value needs to be < max value");
    assertm(scale_val > 0.0, "Scale value must be greater than zero.");
    psi_min_val = min_val;
    psi_max_val = max_val;
    psi_scale = scale_val;
}

void PsiPhiArray::set_phi_scaling(float min_val, float max_val, float scale_val) {
    assertm(min_val < max_val, "Min value needs to be < max value");
    assertm(scale_val > 0.0, "Scale value must be greater than zero.");
    phi_min_val = min_val;
    phi_max_val = max_val;
    phi_scale = scale_val;
}

// -------------------------------------------
// --- Implementation of utility functions ---
// -------------------------------------------

// Compute the min, max, and scale parameter from the a vector of image data.
std::array<float, 3>  compute_scale_params_from_image_vect(const std::vector<RawImage>& imgs, int num_bytes) {
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
    assertm(data.get_cpu_array_ptr() != nullptr, "CPU PsiPhi already allocated.");
    T *encoded = (T *)malloc(data.get_total_array_size());

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
                    psi_value = encode_uint_scalar(psi_value, data.get_psi_min_val(), safe_max_psi, data.get_psi_scale());
                    phi_value = encode_uint_scalar(phi_value, data.get_phi_min_val(), safe_max_phi, data.get_phi_scale());
                }

                encoded[current_index++] = static_cast<T>(psi_value);
                encoded[current_index++] = static_cast<T>(phi_value);
                assertm(current_index <= data.get_num_entries(), "Out of bounds write.");
            }
        }
    }

    data.set_cpu_array_ptr((void*)encoded);
}

void fill_psi_phi_array(PsiPhiArray& result_data,
                        const std::vector<RawImage>& psi_imgs, 
                        const std::vector<RawImage>& phi_imgs) {
    if (result_data.get_cpu_array_ptr() != nullptr) {
        return;
    }

    // Set the meta data and do a bunch of validity checks.
    int num_times = psi_imgs.size();
    assertm(num_times > 0, "Trying to fill PsiPhi from empty vectors.");
    assertm(num_times = phi_imgs.size(), "Size mismatch between psi and phi.");

    int width = phi_imgs[0].get_width();
    int height = phi_imgs[0].get_height();
    result_data.set_meta_data(num_times, width, height);

    // Compute the scaling parameters.
    std::array<float, 3> psi_params = compute_scale_params_from_image_vect(psi_imgs, result_data.get_num_bytes());
    result_data.set_psi_scaling(psi_params[0], psi_params[1], psi_params[2]);
 
    std::array<float, 3> phi_params = compute_scale_params_from_image_vect(phi_imgs, result_data.get_num_bytes());
    result_data.set_phi_scaling(phi_params[0], phi_params[1], phi_params[2]);

    // Compute the memory footprint for the arrays and do the local encoding.
    
    if (result_data.get_num_bytes() == 1) {
        set_encode_cpu_psi_phi_array<uint8_t>(result_data, psi_imgs, phi_imgs);
    } else if (result_data.get_num_bytes() == 2) {      
        set_encode_cpu_psi_phi_array<uint16_t>(result_data, psi_imgs, phi_imgs);
    } else {    
        set_encode_cpu_psi_phi_array<float>(result_data, psi_imgs, phi_imgs);
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

    py::class_<search::PsiPhi>(m, "PsiPhi")
            .def(py::init<>())
            .def_readwrite("psi", &search::PsiPhi::psi)
            .def_readwrite("phi", &search::PsiPhi::phi);

    py::class_<ppa>(m, "PsiPhiArray")
        .def(py::init<>())
        .def(py::init<int>())
        .def_property_readonly("num_bytes", &ppa::get_num_bytes)
        .def_property_readonly("num_times", &ppa::get_num_times)
        .def_property_readonly("width", &ppa::get_width)
        .def_property_readonly("height", &ppa::get_height)
        .def_property_readonly("pixels_per_image", &ppa::get_pixels_per_image)
        .def_property_readonly("num_entries", &ppa::get_num_entries)
        .def_property_readonly("total_array_size", &ppa::get_total_array_size)
        .def_property_readonly("block_size", &ppa::get_block_size)
        .def_property_readonly("psi_min_val", &ppa::get_psi_min_val)
        .def_property_readonly("psi_max_val", &ppa::get_psi_max_val)
        .def_property_readonly("psi_scale", &ppa::get_psi_scale)
        .def_property_readonly("phi_min_val", &ppa::get_phi_min_val)
        .def_property_readonly("phi_max_val", &ppa::get_phi_max_val)
        .def_property_readonly("phi_scale", &ppa::get_phi_scale)
        .def_property_readonly("cpu_array_allocated", &ppa::cpu_array_allocated)
        .def_property_readonly("gpu_array_allocated", &ppa::gpu_array_allocated)
        .def("set_meta_data", &ppa::set_meta_data)
        .def("clear", &ppa::clear)
        .def("read_encoded_psi_phi", &ppa::read_encoded_psi_phi);
    m.def("compute_scale_params_from_image_vect", &search::compute_scale_params_from_image_vect);
    m.def("decode_uint_scalar", &search::decode_uint_scalar);
    m.def("encode_uint_scalar", &search::encode_uint_scalar);
    m.def("fill_psi_phi_array", &search::fill_psi_phi_array);
}
#endif

} /* namespace search */
