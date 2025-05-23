#include "logging.h"

#include "kernel_testing_helpers.h"
#include "psi_phi_array_ds.h"
#include "psi_phi_array_utils.h"
#include "pydocs/psi_phi_array_docs.h"

// Declaration of CUDA functions that will be linked in.
#ifdef HAVE_CUDA
#include "kernels/kernel_memory.h"
#endif

namespace search {

// -------------------------------------------------------
// --- Implementation of core data structure functions ---
// -------------------------------------------------------

PsiPhiArray::PsiPhiArray() {
    data_on_gpu = false;
    cpu_array_ptr = nullptr;
    gpu_array_ptr = nullptr;
}

PsiPhiArray::~PsiPhiArray() { clear(); }

void PsiPhiArray::clear() {
    // Free all used memory on CPU and GPU.
    if (cpu_array_ptr != nullptr) {
        free(cpu_array_ptr);
        cpu_array_ptr = nullptr;
    }
    cpu_time_array.clear();
    clear_from_gpu();

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

void PsiPhiArray::clear_from_gpu() {
#ifdef HAVE_CUDA
    logging::Logger* logger = logging::getLogger("kbmod.search.psi_phi_array");

    if (gpu_time_array.on_gpu()) {
        logger->debug(stat_gpu_memory_mb());
        logger->debug("Freeing times on GPU. " + gpu_time_array.stats_string());
        gpu_time_array.free_gpu_memory();
    }

    if (gpu_array_ptr != nullptr) {
        logger->debug("Freeing PsiPhiArray on GPU: " + std::to_string(get_total_array_size()) + " bytes");
        free_gpu_block(gpu_array_ptr);
        logger->debug(stat_gpu_memory_mb());
    }
#endif
    gpu_array_ptr = nullptr;
    data_on_gpu = false;
}

void PsiPhiArray::move_to_gpu() {
    if (data_on_gpu) {
        if ((gpu_array_ptr == nullptr) || !gpu_time_array.on_gpu()) {
            throw std::runtime_error("Inconsistent GPU flags and pointers");
        }
        return;  // Nothing to do.
    }
    if (cpu_array_ptr == nullptr) std::runtime_error("CPU data not allocated.");
    if (gpu_array_ptr != nullptr) std::runtime_error("GPU psi/phi already allocated.");
    if (gpu_time_array.on_gpu()) std::runtime_error("GPU time already allocated.");
    assert_sizes_equal(cpu_time_array.size(), meta_data.num_times, "psi-phi number of times");

#ifdef HAVE_CUDA
    logging::Logger* logger = logging::getLogger("kbmod.search.psi_phi_array");

    // Copy the Psi/Phi
    logger->debug(stat_gpu_memory_mb());
    gpu_array_ptr = allocate_gpu_block(get_total_array_size());
    logger->debug("Allocating PsiPhiArray on GPU: " + std::to_string(get_total_array_size() / (1024 * 1024)) +
                  " MB");
    copy_block_to_gpu(cpu_array_ptr, gpu_array_ptr, get_total_array_size());

    // Copy the GPU times.
    gpu_time_array.resize(cpu_time_array.size());
    logger->debug("Allocating times on GPU: " + gpu_time_array.stats_string());
    gpu_time_array.copy_vector_to_gpu(cpu_time_array);
    logger->debug(stat_gpu_memory_mb());

    data_on_gpu = true;
#endif
}

void PsiPhiArray::set_meta_data(int new_num_bytes, uint64_t new_num_times, uint64_t new_height,
                                uint64_t new_width) {
    // Validity checking of parameters.
    if (new_num_bytes != -1 && new_num_bytes != 1 && new_num_bytes != 2 && new_num_bytes != 4) {
        throw std::runtime_error("Invalid setting of num_bytes. Must be (-1 [use default], 1, 2, or 4).");
    }
    if (new_num_times == 0) throw std::runtime_error("Invalid num_times passed to set_meta_data.");
    if (new_width == 0) throw std::runtime_error("Invalid width passed to set_meta_data.");
    if (new_height == 0) throw std::runtime_error("Invalid height passed to set_meta_data.");

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
        meta_data.num_bytes = 4;
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

void PsiPhiArray::set_time_array(const std::vector<double>& times) { cpu_time_array = times; }

PsiPhi PsiPhiArray::read_psi_phi(uint64_t time, int row, int col) const {
    PsiPhi result = {NO_DATA, NO_DATA};

    // Array allocation and bounds checking.
    if ((cpu_array_ptr == nullptr) || (row < 0) || (col < 0) || (row >= meta_data.height) ||
        (col >= meta_data.width)) {
        return result;
    }

    // Compute the in-list index from the row, column, and time.
    uint64_t start_index =
            2 * (meta_data.pixels_per_image * time + static_cast<uint64_t>(row * meta_data.width + col));

    if (meta_data.num_bytes == 4) {
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

double PsiPhiArray::read_time(uint64_t time_index) const {
    if (time_index >= meta_data.num_times) {
        throw std::runtime_error("Out of bounds read for time step. [" + std::to_string(time_index) + "]");
    }
    return cpu_time_array[time_index];
}

// -------------------------------------------
// --- Implementation of utility functions ---
// -------------------------------------------

// Compute the min, max, and scale parameter from the a vector of image data.
std::array<float, 3> compute_scale_params_from_image_vect(const std::vector<Image>& imgs, int num_bytes) {
    int num_images = imgs.size();

    // Do a linear pass through the all the pixels to compute the scaling parameters for psi and phi.
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (int i = 0; i < num_images; ++i) {
        for (auto elem : imgs[i].reshaped()) {
            if (pixel_value_valid(elem)) {
                min_val = std::min(min_val, elem);
                max_val = std::max(max_val, elem);
            }
        }
    }

    // Set the scale if we are encoding the values.
    float scale = 1.0;
    if (num_bytes == 1 || num_bytes == 2) {
        float width = (max_val - min_val);
        if (width < 1e-6) width = 1e-6;  // Avoid a zero width.

        uint64_t num_values = (1 << (8 * num_bytes)) - 1;
        scale = width / (double)num_values;
    }

    return {min_val, max_val, scale};
}

template <typename T>
void set_encode_cpu_psi_phi_array(PsiPhiArray& data, const std::vector<Image>& psi_imgs,
                                  const std::vector<Image>& phi_imgs) {
    if (data.get_cpu_array_ptr() != nullptr) {
        throw std::runtime_error("CPU PsiPhi already allocated.");
    }
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating CPU memory for encoded PsiPhi array using " +
                    std::to_string(data.get_total_array_size()) + " bytes");

    T* encoded = (T*)malloc(data.get_total_array_size());
    if (encoded == nullptr) {
        throw std::runtime_error("Unable to allocate space for CPU PsiPhi array.");
    }

    // Create a safe maximum that is slightly less than the true max to avoid
    // rollover of the unsigned integer.
    float safe_max_psi = data.get_psi_max_val() - data.get_psi_scale() / 100.0;
    float safe_max_phi = data.get_phi_max_val() - data.get_phi_scale() / 100.0;

    // We use a uint64_t to prevent overflow on large image stacks
    uint64_t current_index = 0;
    int num_bytes = data.get_num_bytes();
    for (int t = 0; t < data.get_num_times(); ++t) {
        for (int row = 0; row < data.get_height(); ++row) {
            for (int col = 0; col < data.get_width(); ++col) {
                float psi_value = psi_imgs[t](row, col);
                float phi_value = phi_imgs[t](row, col);

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

void set_float_cpu_psi_phi_array(PsiPhiArray& data, const std::vector<Image>& psi_imgs,
                                 const std::vector<Image>& phi_imgs) {
    if (data.get_cpu_array_ptr() != nullptr) {
        throw std::runtime_error("CPU PsiPhi already allocated.");
    }
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating CPU memory for float PsiPhi array using " +
                    std::to_string(data.get_total_array_size()) + " bytes");

    float* encoded = (float*)malloc(data.get_total_array_size());
    if (encoded == nullptr) {
        throw std::runtime_error("Unable to allocate space for CPU PsiPhi array.");
    }

    // We use a uint64_t to prevent overflow on large image stacks
    uint64_t current_index = 0;
    for (int t = 0; t < data.get_num_times(); ++t) {
        for (int row = 0; row < data.get_height(); ++row) {
            for (int col = 0; col < data.get_width(); ++col) {
                encoded[current_index++] = psi_imgs[t](row, col);
                encoded[current_index++] = phi_imgs[t](row, col);
            }
        }
    }

    data.set_cpu_array_ptr((void*)encoded);
}

void fill_psi_phi_array(PsiPhiArray& result_data, int num_bytes, const std::vector<Image>& psi_imgs,
                        const std::vector<Image>& phi_imgs, const std::vector<double> zeroed_times) {
    if (result_data.get_cpu_array_ptr() != nullptr) {
        return;
    }

    // Set the meta data and do a bunch of validity checks.
    uint64_t num_times = psi_imgs.size();
    if (num_times == 0) throw std::runtime_error("Trying to fill PsiPhi from empty vectors.");
    assert_sizes_equal(phi_imgs.size(), num_times, "psi and phi arrays");
    assert_sizes_equal(phi_imgs.size(), num_times, "psi array and zeroed times");

    uint64_t width = phi_imgs[0].cols();
    uint64_t height = phi_imgs[0].rows();
    result_data.set_meta_data(num_bytes, num_times, height, width);

    if (result_data.get_num_bytes() == 1 || result_data.get_num_bytes() == 2) {
        // Compute the scaling parameters needed for encoding.
        std::array<float, 3> psi_params =
                compute_scale_params_from_image_vect(psi_imgs, result_data.get_num_bytes());
        result_data.set_psi_scaling(psi_params[0], psi_params[1], psi_params[2]);

        std::array<float, 3> phi_params =
                compute_scale_params_from_image_vect(phi_imgs, result_data.get_num_bytes());
        result_data.set_phi_scaling(phi_params[0], phi_params[1], phi_params[2]);

        logging::getLogger("kbmod.search.psi_phi_array")
                ->info("Encoding psi to " + std::to_string(result_data.get_num_bytes()) +
                       ": min=" + std::to_string(psi_params[0]) + ", max=" + std::to_string(psi_params[1]) +
                       ", scale=" + std::to_string(psi_params[2]));
        logging::getLogger("kbmod.search.psi_phi_array")
                ->info("Encoding phi to " + std::to_string(result_data.get_num_bytes()) +
                       ": min=" + std::to_string(phi_params[0]) + ", max=" + std::to_string(phi_params[1]) +
                       ", scale=" + std::to_string(phi_params[2]));

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

    // Copy the time array.
    const uint64_t times_bytes = result_data.get_num_times() * sizeof(double);
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating times on CPU: " + std::to_string(times_bytes) + " bytes.");
    result_data.set_time_array(zeroed_times);
}

void fill_psi_phi_array_from_image_arrays(PsiPhiArray& result_data, int num_bytes,
                                          std::vector<Image>& sci_imgs, std::vector<Image>& var_imgs,
                                          std::vector<Image>& psf_kernels,
                                          std::vector<double>& zeroed_times) {
    const uint64_t num_images = sci_imgs.size();
    if (num_images == 0) {
        throw std::runtime_error("Trying to fill PsiPhi from empty vectors.");
    }
    if (num_images != var_imgs.size()) {
        throw std::runtime_error("Number of images in sci and var do not match.");
    }
    const uint64_t height = sci_imgs[0].rows();
    const uint64_t width = sci_imgs[0].cols();
    const uint64_t total_bytes = 2 * height * width * num_images * sizeof(float);

    logging::getLogger("kbmod.search.psi_phi_array")
            ->info("Building " + std::to_string(num_images * 2) + " temporary " + std::to_string(height) +
                   " by " + std::to_string(width) + " images, requiring " + std::to_string(total_bytes) +
                   " bytes.");

    // Build the psi and phi images first.
    std::vector<Image> psi_images;
    std::vector<Image> phi_images;
    for (uint64_t i = 0; i < num_images; ++i) {
        Image& sci = sci_imgs[i];
        Image& var = var_imgs[i];
        Image& psf = psf_kernels[i];

        psi_images.push_back(generate_psi(sci, var, psf));
        phi_images.push_back(generate_phi(var, psf));
    }

    // Convert these into an array form. Needs the full psi and phi computed first so the
    // encoding can compute the bounds of each array.
    fill_psi_phi_array(result_data, num_bytes, psi_images, phi_images, zeroed_times);
}

// -------------------------------------------
// --- Python definitions --------------------
// -------------------------------------------

#ifdef Py_PYTHON_H
static void psi_phi_array_binding(py::module& m) {
    using ppa = search::PsiPhiArray;

    py::class_<search::PsiPhi>(m, "PsiPhi", pydocs::DOC_PsiPhi)
            .def(py::init<>())
            .def_readwrite("psi", &search::PsiPhi::psi)
            .def_readwrite("phi", &search::PsiPhi::phi);

    py::class_<ppa>(m, "PsiPhiArray", pydocs::DOC_PsiPhiArray)
            .def(py::init<>())
            .def_property_readonly("on_gpu", &ppa::on_gpu, pydocs::DOC_PsiPhiArray_on_gpu)
            .def_property_readonly("num_bytes", &ppa::get_num_bytes, pydocs::DOC_PsiPhiArray_get_num_bytes)
            .def_property_readonly("num_times", &ppa::get_num_times, pydocs::DOC_PsiPhiArray_get_num_times)
            .def_property_readonly("width", &ppa::get_width, pydocs::DOC_PsiPhiArray_get_width)
            .def_property_readonly("height", &ppa::get_height, pydocs::DOC_PsiPhiArray_get_height)
            .def_property_readonly("pixels_per_image", &ppa::get_pixels_per_image,
                                   pydocs::DOC_PsiPhiArray_get_pixels_per_image)
            .def_property_readonly("num_entries", &ppa::get_num_entries,
                                   pydocs::DOC_PsiPhiArray_get_num_entries)
            .def_property_readonly("total_array_size", &ppa::get_total_array_size,
                                   pydocs::DOC_PsiPhiArray_get_total_array_size)
            .def_property_readonly("block_size", &ppa::get_block_size, pydocs::DOC_PsiPhiArray_get_block_size)
            .def_property_readonly("psi_min_val", &ppa::get_psi_min_val,
                                   pydocs::DOC_PsiPhiArray_get_psi_min_val)
            .def_property_readonly("psi_max_val", &ppa::get_psi_max_val,
                                   pydocs::DOC_PsiPhiArray_get_psi_max_val)
            .def_property_readonly("psi_scale", &ppa::get_psi_scale, pydocs::DOC_PsiPhiArray_get_psi_scale)
            .def_property_readonly("phi_min_val", &ppa::get_phi_min_val,
                                   pydocs::DOC_PsiPhiArray_get_phi_min_val)
            .def_property_readonly("phi_max_val", &ppa::get_phi_max_val,
                                   pydocs::DOC_PsiPhiArray_get_phi_max_val)
            .def_property_readonly("phi_scale", &ppa::get_phi_scale, pydocs::DOC_PsiPhiArray_get_phi_scale)
            .def_property_readonly("cpu_array_allocated", &ppa::cpu_array_allocated,
                                   pydocs::DOC_PsiPhiArray_get_cpu_array_allocated)
            .def_property_readonly("gpu_array_allocated", &ppa::gpu_array_allocated,
                                   pydocs::DOC_PsiPhiArray_get_gpu_array_allocated)
            .def("set_meta_data", &ppa::set_meta_data, pydocs::DOC_PsiPhiArray_set_meta_data)
            .def("set_time_array", &ppa::set_time_array, pydocs::DOC_PsiPhiArray_set_time_array)
            .def("move_to_gpu", &ppa::move_to_gpu, pydocs::DOC_PsiPhiArray_move_to_gpu)
            .def("clear", &ppa::clear, pydocs::DOC_PsiPhiArray_clear)
            .def("clear_from_gpu", &ppa::clear_from_gpu, pydocs::DOC_PsiPhiArray_clear_from_gpu)
            .def("read_psi_phi", &ppa::read_psi_phi, pydocs::DOC_PsiPhiArray_read_psi_phi)
            .def("read_time", &ppa::read_time, pydocs::DOC_PsiPhiArray_read_time);
    m.def("compute_scale_params_from_image_vect", &search::compute_scale_params_from_image_vect);
    m.def("decode_uint_scalar", &search::decode_uint_scalar);
    m.def("encode_uint_scalar", &search::encode_uint_scalar);
    m.def("fill_psi_phi_array", &search::fill_psi_phi_array, pydocs::DOC_PsiPhiArray_fill_psi_phi_array);
    m.def("fill_psi_phi_array_from_image_arrays", &search::fill_psi_phi_array_from_image_arrays,
          pydocs::DOC_PsiPhiArray_fill_psi_phi_array_from_image_arrays);
}
#endif

} /* namespace search */
