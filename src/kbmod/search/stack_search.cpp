#include "stack_search.h"

namespace search {
#ifdef HAVE_CUDA
extern "C" void deviceSearchFilter(PsiPhiArray& psi_phi_array, SearchParameters params,
                                   TrajectoryList& trj_to_search, TrajectoryList& results);

extern "C" void evaluateTrajectory(PsiPhiArrayMeta psi_phi_meta, void* psi_phi_vect, double* image_times,
                                   SearchParameters params, Trajectory* candidate);
#endif

StackSearch::StackSearch(ImageStack& imstack) : stack(imstack), results(0), gpu_search_list(0) {
    psi_phi_generated = false;

    // Default The Thresholds.
    params.min_observations = 0;
    params.min_lh = 0.0;

    // Default filtering arguments.
    params.do_sigmag_filter = false;
    params.sgl_L = 0.25;
    params.sgl_H = 0.75;
    params.sigmag_coeff = -1.0;

    // Default the encoding parameters.
    params.encode_num_bytes = -1;

    // Default the results per pixel.
    params.results_per_pixel = 8;

    // Default pixel starting bounds.
    params.x_start_min = 0;
    params.x_start_max = stack.get_width();
    params.y_start_min = 0;
    params.y_start_max = stack.get_height();

    // Get the logger for this module.
    rs_logger = logging::getLogger("kbmod.search.run_search");
}

// --------------------------------------------
// Configuration functions
// --------------------------------------------

void StackSearch::set_min_obs(int new_value) { params.min_observations = new_value; }

void StackSearch::set_min_lh(float new_value) { params.min_lh = new_value; }

void StackSearch::set_results_per_pixel(int new_value) {
    if (new_value <= 0) throw std::runtime_error("Invalid results per pixel.");
    params.results_per_pixel = new_value;
}

void StackSearch::enable_gpu_sigmag_filter(std::vector<float> percentiles, float sigmag_coeff, float min_lh) {
    if ((percentiles.size() != 2) || (percentiles[0] >= percentiles[1]) || (percentiles[0] <= 0.0) ||
        (percentiles[1] >= 1.0)) {
        throw std::runtime_error("Invalid percentiles for sigma G filtering.");
    }
    if (sigmag_coeff <= 0.0) {
        throw std::runtime_error("Invalid coefficient for sigma G filtering.");
    }

    params.do_sigmag_filter = true;
    params.sgl_L = percentiles[0];
    params.sgl_H = percentiles[1];
    params.sigmag_coeff = sigmag_coeff;
    params.min_lh = min_lh;
}

void StackSearch::enable_gpu_encoding(int encode_num_bytes) {
    // If changing a setting that would impact the search data encoding, clear the cached values.
    if (params.encode_num_bytes != encode_num_bytes) {
        clear_psi_phi();
    }

    // Make sure the encoding is one of the supported options.
    // Otherwise use default float (aka no encoding).
    if (encode_num_bytes == 1 || encode_num_bytes == 2) {
        params.encode_num_bytes = encode_num_bytes;
    } else {
        params.encode_num_bytes = -1;
    }
}

void StackSearch::set_start_bounds_x(int x_min, int x_max) {
    if (x_min >= x_max) {
        throw std::runtime_error("Invalid search bounds for the x pixel [" + std::to_string(x_min) +
                                 ", " + std::to_string(x_max) + "]");
    }
    params.x_start_min = x_min;
    params.x_start_max = x_max;
}

void StackSearch::set_start_bounds_y(int y_min, int y_max) {
    if (y_min >= y_max) {
        throw std::runtime_error("Invalid search bounds for the y pixel [" + std::to_string(y_min) +
                                 ", " + std::to_string(y_max) + "]");
    }
    params.y_start_min = y_min;
    params.y_start_max = y_max;
}

// --------------------------------------------
// Data precomputation functions
// --------------------------------------------

void StackSearch::prepare_psi_phi() {
    if (!psi_phi_generated) {
        DebugTimer timer = DebugTimer("preparing Psi and Phi images", rs_logger);
        fill_psi_phi_array_from_image_stack(psi_phi_array, stack, params.encode_num_bytes);
        timer.stop();
        psi_phi_generated = true;
    }

    // Perform additional error checking that the arrays are allocated (checked even if
    // using the cached values).
    if (!psi_phi_array.cpu_array_allocated()) {
        throw std::runtime_error("PsiPhiArray array unallocated after prepare_psi_phi_array.");
    }
}

void StackSearch::clear_psi_phi() {
    if (psi_phi_generated) {
        psi_phi_array.clear();
        psi_phi_generated = false;
    }
}

// --------------------------------------------
// Core search functions
// --------------------------------------------

void StackSearch::evaluate_single_trajectory(Trajectory& trj) {
    prepare_psi_phi();
    if (!psi_phi_array.cpu_array_allocated()) std::runtime_error("Data not allocated.");

#ifdef HAVE_CUDA
    evaluateTrajectory(psi_phi_array.get_meta_data(), psi_phi_array.get_cpu_array_ptr(),
                       psi_phi_array.get_cpu_time_array_ptr(), params, &trj);
#else
    throw std::runtime_error("CUDA installation is needed for single trajectory search.");
#endif
}

Trajectory StackSearch::search_linear_trajectory(int x, int y, float vx, float vy) {
    Trajectory result;
    result.x = x;
    result.y = y;
    result.vx = vx;
    result.vy = vy;

    evaluate_single_trajectory(result);

    return result;
}

void StackSearch::finish_search() {
    psi_phi_array.clear_from_gpu();
    gpu_search_list.move_to_cpu();
}

void StackSearch::prepare_search(std::vector<Trajectory>& search_list, int min_observations) {
    DebugTimer psi_phi_timer = DebugTimer("Creating psi/phi buffers", rs_logger);
    prepare_psi_phi();
    psi_phi_array.move_to_gpu();
    psi_phi_timer.stop();

    uint64_t num_to_search = search_list.size();

    rs_logger->info("Preparing to search " + std::to_string(num_to_search) + " trajectories.");
    gpu_search_list.set_trajectories(search_list);
    gpu_search_list.move_to_gpu();

    params.min_observations = min_observations;
}

void StackSearch::search_all(std::vector<Trajectory>& search_list, int min_observations) {
    prepare_search(search_list, min_observations);
    search_batch();
    finish_search();
}

void StackSearch::search_batch() {
    if (!psi_phi_array.gpu_array_allocated()) {
        throw std::runtime_error(
                "PsiPhiArray array not allocated on GPU. Did you forget to call prepare_search?");
    }

    DebugTimer core_timer = DebugTimer("Running batch search", rs_logger);
    uint64_t max_results = compute_max_results();

    // staple C++
    std::stringstream logmsg;
    logmsg << "Searching X=[" << params.x_start_min << ", " << params.x_start_max << "] "
           << "Y=[" << params.y_start_min << ", " << params.y_start_max << "]\n"
           << "Allocating space for " << max_results << " results.";
    rs_logger->info(logmsg.str());

    results.resize(max_results);
    results.move_to_gpu();

    // Do the actual search on the GPU.
    DebugTimer search_timer = DebugTimer("Running search", rs_logger);
#ifdef HAVE_CUDA
    deviceSearchFilter(psi_phi_array, params, gpu_search_list, results);
#else
    throw std::runtime_error("Non-GPU search is not implemented.");
#endif
    search_timer.stop();

    results.move_to_cpu();
    DebugTimer sort_timer = DebugTimer("Sorting results", rs_logger);
    results.sort_by_likelihood();
    sort_timer.stop();
    core_timer.stop();
}

std::vector<Trajectory> StackSearch::search_single_batch() {
    uint64_t max_results = compute_max_results();
    search_batch();
    return results.get_batch(0, max_results);
}

uint64_t StackSearch::compute_max_results() {
    uint64_t search_width = params.x_start_max - params.x_start_min;
    uint64_t search_height = params.y_start_max - params.y_start_min;
    uint64_t num_search_pixels = search_width * search_height;
    return num_search_pixels * params.results_per_pixel;
}

std::vector<float> StackSearch::extract_psi_or_phi_curve(Trajectory& trj, bool extract_psi) {
    prepare_psi_phi();

    const unsigned int num_times = stack.img_count();
    std::vector<float> result(num_times, 0.0);

    for (unsigned int i = 0; i < num_times; ++i) {
        double time = psi_phi_array.read_time(i);

        // Query the center of the predicted location's pixel.
        PsiPhi psi_phi_val = psi_phi_array.read_psi_phi(i, trj.get_y_index(time), trj.get_x_index(time));
        float value = (extract_psi) ? psi_phi_val.psi : psi_phi_val.phi;
        if (pixel_value_valid(value)) {
            result[i] = value;
        }
    }
    return result;
}

std::vector<float> StackSearch::get_psi_curves(Trajectory& trj) {
    return extract_psi_or_phi_curve(trj, true);
}

std::vector<float> StackSearch::get_phi_curves(Trajectory& trj) {
    return extract_psi_or_phi_curve(trj, false);
}

std::vector<Trajectory> StackSearch::get_results(uint64_t start, uint64_t count) {
    return results.get_batch(start, count);
}

// This function is used only for testing by injecting known result trajectories.
void StackSearch::set_results(const std::vector<Trajectory>& new_results) {
    results.set_trajectories(new_results);
}

#ifdef Py_PYTHON_H
static void stack_search_bindings(py::module& m) {
    using tj = search::Trajectory;
    using pf = search::PSF;
    using ri = search::RawImage;
    using is = search::ImageStack;
    using ks = search::StackSearch;

    py::class_<ks>(m, "StackSearch", pydocs::DOC_StackSearch)
            .def(py::init<is&>())
            .def("search_all", &ks::search_all, pydocs::DOC_StackSearch_search)
            .def("evaluate_single_trajectory", &ks::evaluate_single_trajectory,
                 pydocs::DOC_StackSearch_evaluate_single_trajectory)
            .def("search_linear_trajectory", &ks::search_linear_trajectory,
                 pydocs::DOC_StackSearch_search_linear_trajectory)
            .def("set_min_obs", &ks::set_min_obs, pydocs::DOC_StackSearch_set_min_obs)
            .def("set_min_lh", &ks::set_min_lh, pydocs::DOC_StackSearch_set_min_lh)
            .def("set_results_per_pixel", &ks::set_results_per_pixel,
                 pydocs::DOC_StackSearch_set_results_per_pixel)
            .def("enable_gpu_sigmag_filter", &ks::enable_gpu_sigmag_filter,
                 pydocs::DOC_StackSearch_enable_gpu_sigmag_filter)
            .def("enable_gpu_encoding", &ks::enable_gpu_encoding, pydocs::DOC_StackSearch_enable_gpu_encoding)
            .def("set_start_bounds_x", &ks::set_start_bounds_x, pydocs::DOC_StackSearch_set_start_bounds_x)
            .def("set_start_bounds_y", &ks::set_start_bounds_y, pydocs::DOC_StackSearch_set_start_bounds_y)
            .def("get_num_images", &ks::num_images, pydocs::DOC_StackSearch_get_num_images)
            .def("get_image_width", &ks::get_image_width, pydocs::DOC_StackSearch_get_image_width)
            .def("get_image_height", &ks::get_image_height, pydocs::DOC_StackSearch_get_image_height)
            .def("get_image_npixels", &ks::get_image_npixels, pydocs::DOC_StackSearch_get_image_npixels)
            .def("get_imagestack", &ks::get_imagestack, py::return_value_policy::reference_internal,
                 pydocs::DOC_StackSearch_get_imagestack)
            // For testings
            .def("get_psi_curves", (std::vector<float>(ks::*)(tj&)) & ks::get_psi_curves,
                 pydocs::DOC_StackSearch_get_psi_curves)
            .def("get_phi_curves", (std::vector<float>(ks::*)(tj&)) & ks::get_phi_curves,
                 pydocs::DOC_StackSearch_get_phi_curves)
            .def("prepare_psi_phi", &ks::prepare_psi_phi, pydocs::DOC_StackSearch_prepare_psi_phi)
            .def("clear_psi_phi", &ks::clear_psi_phi, pydocs::DOC_StackSearch_clear_psi_phi)
            .def("get_results", &ks::get_results, pydocs::DOC_StackSearch_get_results)
            .def("set_results", &ks::set_results, pydocs::DOC_StackSearch_set_results)
            .def("compute_max_results", &ks::compute_max_results, pydocs::DOC_StackSearch_compute_max_results)
            .def("search_single_batch", &ks::search_single_batch, pydocs::DOC_StackSearch_search_single_batch)
            .def("prepare_search", &ks::prepare_search, pydocs::DOC_StackSearch_prepare_batch_search)
            .def("finish_search", &ks::finish_search, pydocs::DOC_StackSearch_finish_search);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
