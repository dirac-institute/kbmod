#include "logging.h"

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
}

void PsiPhiArray::clear_from_gpu() {
    if (!data_on_gpu) {
        if ((gpu_array_ptr != nullptr) || gpu_time_array.on_gpu()) {
            throw std::runtime_error("Inconsistent GPU flags and pointers");
        }
        return;  // Nothing to do.
    }
    if ((gpu_array_ptr == nullptr) || !gpu_time_array.on_gpu()) {
        throw std::runtime_error("Inconsistent GPU flags and pointers");
    }

#ifdef HAVE_CUDA
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Freeing times on GPU. " + gpu_time_array.stats_string());
    gpu_time_array.free_gpu_memory();

    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Freeing PsiPhiArray on GPU: " + std::to_string(get_total_array_size()) + " bytes");
    free_gpu_block(gpu_array_ptr);
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
    // Copy the Psi/Phi data
    gpu_array_ptr = allocate_gpu_block(get_total_array_size());
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating PsiPhiArray on GPU: " +
                    std::to_string(get_total_array_size() / (1024 * 1024)) + " MB");
    copy_block_to_gpu(cpu_array_ptr, gpu_array_ptr, get_total_array_size());

    // Copy the GPU times.
    gpu_time_array.resize(cpu_time_array.size());
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating times on GPU: " + gpu_time_array.stats_string());
    gpu_time_array.copy_vector_to_gpu(cpu_time_array);

    data_on_gpu = true;
#endif
}

void PsiPhiArray::set_meta_data(uint64_t new_num_times, uint64_t new_height, uint64_t new_width) {
    // Validity checking of parameters.
    if (new_num_times == 0) throw std::runtime_error("Invalid num_times passed to set_meta_data.");
    if (new_width == 0) throw std::runtime_error("Invalid width passed to set_meta_data.");
    if (new_height == 0) throw std::runtime_error("Invalid height passed to set_meta_data.");

    // Check that we do not have an array allocated.
    if (cpu_array_ptr != nullptr) {
        throw std::runtime_error("Cannot change meta data with allocated arrays. Call clear() first.");
    }

    meta_data.block_size = sizeof(float);
    meta_data.num_times = new_num_times;
    meta_data.width = new_width;
    meta_data.height = new_height;
    meta_data.pixels_per_image = meta_data.width * meta_data.height;
    meta_data.num_entries = 2 * meta_data.pixels_per_image * meta_data.num_times;
    meta_data.total_array_size = meta_data.block_size * meta_data.num_entries;
}

void PsiPhiArray::set_time_array(const std::vector<double>& times) { cpu_time_array = times; }

PsiPhi PsiPhiArray::read_psi_phi(uint64_t time, int row, int col) {
    PsiPhi result = {NO_DATA, NO_DATA};

    // Array allocation and bounds checking.
    if ((cpu_array_ptr == nullptr) || (row < 0) || (col < 0) || (row >= meta_data.height) ||
        (col >= meta_data.width)) {
        return result;
    }

    // Compute the in-list index from the row, column, and time.
    uint64_t start_index =
            2 * (meta_data.pixels_per_image * time + static_cast<uint64_t>(row * meta_data.width + col));
    result.psi = reinterpret_cast<float*>(cpu_array_ptr)[start_index];
    result.phi = reinterpret_cast<float*>(cpu_array_ptr)[start_index + 1];
    return result;
}

double PsiPhiArray::read_time(uint64_t time_index) {
    if (time_index >= meta_data.num_times) {
        throw std::runtime_error("Out of bounds read for time step. [" + std::to_string(time_index) + "]");
    }
    return cpu_time_array[time_index];
}

// -------------------------------------------
// --- Implementation of utility functions ---
// -------------------------------------------

void fill_psi_phi_array(PsiPhiArray& result_data, const std::vector<RawImage>& psi_imgs,
                        const std::vector<RawImage>& phi_imgs, const std::vector<double> zeroed_times) {
    if (result_data.get_cpu_array_ptr() != nullptr) {
        return;
    }

    // Set the meta data and do a bunch of validity checks.
    uint64_t num_times = psi_imgs.size();
    if (num_times == 0) throw std::runtime_error("Trying to fill PsiPhi from empty vectors.");
    assert_sizes_equal(phi_imgs.size(), num_times, "psi and phi arrays");
    assert_sizes_equal(phi_imgs.size(), num_times, "psi array and zeroed times");

    uint64_t width = phi_imgs[0].get_width();
    uint64_t height = phi_imgs[0].get_height();
    result_data.set_meta_data(num_times, height, width);

    // Create an array of interleaved psi and phi images.
    uint64_t total_bytes = result_data.get_total_array_size();
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating CPU memory for float PsiPhi array using " + std::to_string(total_bytes) +
                    " bytes");
    float* encoded = (float*)malloc(total_bytes);
    if (encoded == nullptr) {
        throw std::runtime_error("Unable to allocate space for CPU PsiPhi array.");
    }

    // We use a uint64_t to prevent overflow on large image stacks
    uint64_t current_index = 0;
    for (int t = 0; t < num_times; ++t) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                encoded[current_index++] = psi_imgs[t].get_pixel({row, col});
                encoded[current_index++] = phi_imgs[t].get_pixel({row, col});
            }
        }
    }
    result_data.set_cpu_array_ptr((void*)encoded);

    // Copy the time array.
    const uint64_t times_bytes = result_data.get_num_times() * sizeof(double);
    logging::getLogger("kbmod.search.psi_phi_array")
            ->debug("Allocating times on CPU: " + std::to_string(times_bytes) + " bytes.");
    result_data.set_time_array(zeroed_times);
}

void fill_psi_phi_array_from_image_stack(PsiPhiArray& result_data, ImageStack& stack) {
    // Compute Phi and Psi from convolved images while leaving masked pixels alone
    // Reinsert 0s for NO_DATA?
    std::vector<RawImage> psi_images;
    std::vector<RawImage> phi_images;
    const uint64_t num_images = stack.img_count();

    uint64_t total_bytes = 2 * stack.get_height() * stack.get_width() * num_images * sizeof(float);
    logging::getLogger("kbmod.search.psi_phi_array")
            ->info("Building " + std::to_string(num_images * 2) + " temporary " +
                   std::to_string(stack.get_height()) + " by " + std::to_string(stack.get_width()) +
                   " images, requiring " + std::to_string(total_bytes) + " bytes.");

    // Build the psi and phi images first.
    for (uint64_t i = 0; i < num_images; ++i) {
        LayeredImage& img = stack.get_single_image(i);
        psi_images.push_back(img.generate_psi_image());
        phi_images.push_back(img.generate_phi_image());
    }

    // Convert these into an array form. Needs the full psi and phi computed first so the
    // encoding can compute the bounds of each array.
    std::vector<double> zeroed_times = stack.build_zeroed_times();
    fill_psi_phi_array(result_data, psi_images, phi_images, zeroed_times);
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
    m.def("fill_psi_phi_array", &search::fill_psi_phi_array, pydocs::DOC_PsiPhiArray_fill_psi_phi_array);
    m.def("fill_psi_phi_array_from_image_stack", &search::fill_psi_phi_array_from_image_stack,
          pydocs::DOC_PsiPhiArray_fill_psi_phi_array_from_image_stack);
}
#endif

} /* namespace search */
