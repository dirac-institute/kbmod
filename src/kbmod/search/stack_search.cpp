#include "stack_search.h"

namespace search {
#ifdef HAVE_CUDA
extern "C" void deviceSearchFilter(PsiPhiArray& psi_phi_array, SearchParameters params,
                                   TrajectoryList& trj_to_search, TrajectoryList& results);

extern "C" void evaluateTrajectory(PsiPhiArrayMeta psi_phi_meta, void* psi_phi_vect, float* image_times,
                                   SearchParameters params, Trajectory* candidate);
#endif

// This logger is often used in this module so we might as well declare it
// global, but this would generally be a one-liner like:
// logging::getLogger("kbmod.search.run_search") -> level(msg)
// I'd imaging...
auto rs_logger = logging::getLogger("kbmod.search.run_search");

StackSearch::StackSearch(ImageStack& imstack) : stack(imstack), results(0) {
    debug_info = false;
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

    // Default pixel starting bounds.
    params.x_start_min = 0;
    params.x_start_max = stack.get_width();
    params.y_start_min = 0;
    params.y_start_max = stack.get_height();

    params.debug = false;
}

// --------------------------------------------
// Configuration functions
// --------------------------------------------

void StackSearch::set_debug(bool d) {
    debug_info = d;
    params.debug = d;
}

void StackSearch::set_min_obs(int new_value) { params.min_observations = new_value; }

void StackSearch::set_min_lh(float new_value) { params.min_lh = new_value; }

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
        throw std::runtime_error("Invalid search bounds for the x pixel.");
    }
    params.x_start_min = x_min;
    params.x_start_max = x_max;
}

void StackSearch::set_start_bounds_y(int y_min, int y_max) {
    if (y_min >= y_max) {
        throw std::runtime_error("Invalid search bounds for the y pixel.");
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
        fill_psi_phi_array_from_image_stack(psi_phi_array, stack, params.encode_num_bytes, debug_info);
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

Trajectory StackSearch::search_linear_trajectory(short x, short y, float vx, float vy) {
    Trajectory result;
    result.x = x;
    result.y = y;
    result.vx = vx;
    result.vy = vy;

    evaluate_single_trajectory(result);

    return result;
}

void StackSearch::search(std::vector<Trajectory>& search_list, int min_observations) {
    DebugTimer core_timer = DebugTimer("core search", rs_logger);

    DebugTimer psi_phi_timer = DebugTimer("creating psi/phi buffers", rs_logger);
    prepare_psi_phi();
    psi_phi_array.move_to_gpu();
    psi_phi_timer.stop();

    // Allocate a vector for the results and move it onto the GPU.
    int search_width = params.x_start_max - params.x_start_min;
    int search_height = params.y_start_max - params.y_start_min;
    int num_search_pixels = search_width * search_height;
    int max_results = num_search_pixels * RESULTS_PER_PIXEL;
    // staple C++
    std::stringstream logmsg;
    logmsg << "Searching X=[" << params.x_start_min << ", " << params.x_start_max << "] "
           << "Y=[" << params.y_start_min << ", " << params.y_start_max << "]\n"
           << "Allocating space for " << max_results << " results.";
    rs_logger->info(logmsg.str());

    results.resize(max_results);
    results.move_to_gpu();

    // Allocate space for the search list and move that to the GPU.
    int num_to_search = search_list.size();

    logmsg.str("");
    logmsg << search_list.size() << " trajectories...";
    rs_logger->info(logmsg.str());

    TrajectoryList gpu_search_list(search_list);
    gpu_search_list.move_to_gpu();

    // Set the minimum number of observations.
    params.min_observations = min_observations;

    // Do the actual search on the GPU.
    DebugTimer search_timer = DebugTimer("search execution", rs_logger);
#ifdef HAVE_CUDA
    deviceSearchFilter(psi_phi_array, params, gpu_search_list, results);
#else
    throw std::runtime_error("Non-GPU search is not implemented.");
#endif
    search_timer.stop();

    // Move data back to CPU to unallocate GPU space (this will happen automatically
    // for gpu_search_list when the object goes out of scope, but we do it explicitly here).
    psi_phi_array.clear_from_gpu();
    results.move_to_cpu();
    gpu_search_list.move_to_cpu();

    DebugTimer sort_timer = DebugTimer("Sorting results", rs_logger);
    results.sort_by_likelihood();
    sort_timer.stop();
    core_timer.stop();
}

std::vector<float> StackSearch::extract_psi_or_phi_curve(Trajectory& trj, bool extract_psi) {
    prepare_psi_phi();

    const int num_times = stack.img_count();
    std::vector<float> result(num_times, 0.0);

    for (int i = 0; i < num_times; ++i) {
        float time = psi_phi_array.read_time(i);

        // Query the center of the predicted location's pixel.
        Point pred_pt = {trj.get_x_pos(time) + 0.5f, trj.get_y_pos(time) + 0.5f};
        Index pred_idx = pred_pt.to_index();
        PsiPhi psi_phi_val = psi_phi_array.read_psi_phi(i, pred_idx.i, pred_idx.j);

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

std::vector<Trajectory> StackSearch::get_results(int start, int count) {
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
            .def("search", &ks::search, pydocs::DOC_StackSearch_search)
            .def("evaluate_single_trajectory", &ks::evaluate_single_trajectory,
                 pydocs::DOC_StackSearch_evaluate_single_trajectory)
            .def("search_linear_trajectory", &ks::search_linear_trajectory,
                 pydocs::DOC_StackSearch_search_linear_trajectory)
            .def("set_min_obs", &ks::set_min_obs, pydocs::DOC_StackSearch_set_min_obs)
            .def("set_min_lh", &ks::set_min_lh, pydocs::DOC_StackSearch_set_min_lh)
            .def("enable_gpu_sigmag_filter", &ks::enable_gpu_sigmag_filter,
                 pydocs::DOC_StackSearch_enable_gpu_sigmag_filter)
            .def("enable_gpu_encoding", &ks::enable_gpu_encoding, pydocs::DOC_StackSearch_enable_gpu_encoding)
            .def("set_start_bounds_x", &ks::set_start_bounds_x, pydocs::DOC_StackSearch_set_start_bounds_x)
            .def("set_start_bounds_y", &ks::set_start_bounds_y, pydocs::DOC_StackSearch_set_start_bounds_y)
            .def("set_debug", &ks::set_debug, pydocs::DOC_StackSearch_set_debug)
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
            .def("set_results", &ks::set_results, pydocs::DOC_StackSearch_set_results);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
