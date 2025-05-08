#include "stack_search.h"

namespace search {
#ifdef HAVE_CUDA
extern "C" void deviceSearchFilter(PsiPhiArray& psi_phi_array, SearchParameters params,
                                   TrajectoryList& trj_to_search, TrajectoryList& results);

extern "C" void evaluateTrajectory(PsiPhiArrayMeta psi_phi_meta, void* psi_phi_vect, double* image_times,
                                   SearchParameters params, Trajectory* candidate);
#endif

// A helper function to extact both the psi and phi information as a single
// list with all psi values and then all phi values.
std::vector<float> extract_joint_psi_phi_curve(const PsiPhiArray& psi_phi, const Trajectory& trj) {
    const unsigned int num_times = psi_phi.get_num_times();
    std::vector<float> result(2 * num_times, 0.0);

    for (unsigned int i = 0; i < num_times; ++i) {
        double time = psi_phi.read_time(i);

        // Query the center of the predicted location's pixel.
        PsiPhi psi_phi_val = psi_phi.read_psi_phi(i, trj.get_y_index(time), trj.get_x_index(time));
        if (pixel_value_valid(psi_phi_val.psi)) {
            result[i] = psi_phi_val.psi;
        }
        if (pixel_value_valid(psi_phi_val.phi)) {
            result[i + num_times] = psi_phi_val.phi;
        }
    }
    return result;
}

// --------------------------------------------
// StackSearch
// --------------------------------------------

StackSearch::StackSearch(ImageStack& imstack) : stack(imstack), results(0) {
    psi_phi_generated = false;

    // Get the logger for this module.
    rs_logger = logging::getLogger("kbmod.search.run_search");

    // Get the cached stats for the images.
    width = imstack.get_width();
    height = imstack.get_height();
    num_imgs = imstack.img_count();
    zeroed_times = imstack.build_zeroed_times();

    set_default_parameters();
}

StackSearch::~StackSearch() {
    // Clear the memory allocated for psi and phi.
    clear_psi_phi();
}

// --------------------------------------------
// Configuration functions
// --------------------------------------------

void StackSearch::set_default_parameters() {
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
    params.x_start_max = width;
    params.y_start_min = 0;
    params.y_start_max = height;
}

void StackSearch::set_min_obs(int new_value) {
    if (new_value < 0) throw std::runtime_error("min_obs must be >= 0.");
    if (new_value > num_imgs)
        throw std::runtime_error("min_obs cannot be greater than the number of images.");

    params.min_observations = new_value;
}

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

void StackSearch::disable_gpu_sigmag_filter() { params.do_sigmag_filter = false; }

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
        throw std::runtime_error("Invalid search bounds for the x pixel [" + std::to_string(x_min) + ", " +
                                 std::to_string(x_max) + "]");
    }
    params.x_start_min = x_min;
    params.x_start_max = x_max;
}

void StackSearch::set_start_bounds_y(int y_min, int y_max) {
    if (y_min >= y_max) {
        throw std::runtime_error("Invalid search bounds for the y pixel [" + std::to_string(y_min) + ", " +
                                 std::to_string(y_max) + "]");
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

void StackSearch::evaluate_single_trajectory(Trajectory& trj, bool use_kernel) {
    prepare_psi_phi();
    if (!psi_phi_array.cpu_array_allocated()) std::runtime_error("Data not allocated.");

    if (!use_kernel) {
        evaluate_trajectory_cpu(psi_phi_array, trj);
    } else {
#ifdef HAVE_CUDA
        if (psi_phi_array.get_num_times() >= MAX_NUM_IMAGES) {
            throw std::runtime_error("Too many images to evaluate on GPU. Max = " +
                                     std::to_string(MAX_NUM_IMAGES));
        }

        evaluateTrajectory(psi_phi_array.get_meta_data(), psi_phi_array.get_cpu_array_ptr(),
                           psi_phi_array.get_cpu_time_array_ptr(), params, &trj);
#else
        throw std::runtime_error("CUDA installation is needed for using kernel code.");
#endif
    }
}

Trajectory StackSearch::search_linear_trajectory(int x, int y, float vx, float vy, bool use_kernel) {
    Trajectory result;
    result.x = x;
    result.y = y;
    result.vx = vx;
    result.vy = vy;

    evaluate_single_trajectory(result, use_kernel);

    return result;
}

void StackSearch::search_all(std::vector<Trajectory>& search_list, bool on_gpu) {
    // Prepare the input data (psi/phi and candidate lists).
    prepare_psi_phi();
    TrajectoryList candidate_list = TrajectoryList(search_list);
    uint64_t max_results = compute_max_results();

    DebugTimer core_timer = DebugTimer("Running batch search", rs_logger);

    // staple C++
    std::stringstream logmsg;
    logmsg << "Searching X=[" << params.x_start_min << ", " << params.x_start_max << "] "
           << "Y=[" << params.y_start_min << ", " << params.y_start_max << "]\n"
           << "Allocating space for " << max_results << " results.";
    rs_logger->info(logmsg.str());
    results.resize(max_results);

    DebugTimer search_timer = DebugTimer("Running search", rs_logger);
    if (on_gpu) {
        // Moved the needed data to the GPU.
        psi_phi_array.move_to_gpu();
        candidate_list.move_to_gpu();
        results.move_to_gpu();

        // Do the actual search on the GPU.
#ifdef HAVE_CUDA
        deviceSearchFilter(psi_phi_array, params, candidate_list, results);
#else
        throw std::runtime_error("Non-GPU search is not implemented.");
#endif

        // Free up the GPU memory.
        results.move_to_cpu();
        candidate_list.move_to_cpu();
        psi_phi_array.clear_from_gpu();
    } else {
        search_cpu_only(psi_phi_array, params, candidate_list, results);
    }
    search_timer.stop();
    uint64_t num_results = results.get_size();
    rs_logger->debug("Core search returned " + std::to_string(num_results) + " results.\n");

    // Perform initial LH and obscount filtering.

    DebugTimer filter_timer = DebugTimer("Filtering results by LH and min_obs", rs_logger);
    results.filter_by_likelihood(params.min_lh);
    results.filter_by_obs_count(params.min_observations);
    uint64_t new_num_results = results.get_size();
    rs_logger->debug("After filtering by LH and min_obs " + std::to_string(new_num_results) + " results (" +
                     std::to_string(num_results - new_num_results) + " removed).\n");
    filter_timer.stop();

    // Sort the results by decreasing likleihood.
    DebugTimer sort_timer = DebugTimer("Sorting results", rs_logger);
    results.sort_by_likelihood();
    sort_timer.stop();

    core_timer.stop();
}

uint64_t StackSearch::compute_max_results() {
    if (params.x_start_min >= params.x_start_max)
        throw std::runtime_error("Invalid search bounds for the x pixel [" +
                                 std::to_string(params.x_start_min) + ", " +
                                 std::to_string(params.x_start_max) + "]");
    if (params.y_start_min >= params.y_start_max)
        throw std::runtime_error("Invalid search bounds for the y pixel [" +
                                 std::to_string(params.y_start_min) + ", " +
                                 std::to_string(params.y_start_max) + "]");

    uint64_t search_width = params.x_start_max - params.x_start_min;
    uint64_t search_height = params.y_start_max - params.y_start_min;
    uint64_t num_search_pixels = search_width * search_height;
    return num_search_pixels * params.results_per_pixel;
}

Image StackSearch::get_all_psi_phi_curves(const std::vector<Trajectory>& trajectories) {
    // Allocate a (num_trj, 2 * num_times) image to store the curves for all the trajectories.
    const unsigned int num_trj = trajectories.size();
    Image results = Image::Zero(num_trj, 2 * num_imgs);

    prepare_psi_phi();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_trj; ++i) {
        std::vector<float> curve = extract_joint_psi_phi_curve(psi_phi_array, trajectories[i]);

// Copy the data into the results.
#pragma omp critical
        for (int j = 0; j < 2 * num_imgs; ++j) {
            results(i, j) = curve[j];
        }
    }
    return results;
}

std::vector<Trajectory> StackSearch::get_results(uint64_t start, uint64_t count) {
    rs_logger->debug("Reading results [" + std::to_string(start) + ", " + std::to_string(start + count) +
                     ")");
    return results.get_batch(start, count);
}

std::vector<Trajectory>& StackSearch::get_all_results() { return results.get_list(); }

// This function is used only for testing by injecting known result trajectories.
void StackSearch::set_results(const std::vector<Trajectory>& new_results) {
    results.set_trajectories(new_results);
}

void StackSearch::clear_results() {
    if (results.on_gpu()) {
        results.move_to_cpu();
    }
    results.resize(0);
}

#ifdef Py_PYTHON_H
static void stack_search_bindings(py::module& m) {
    using tj = search::Trajectory;
    using is = search::ImageStack;
    using ks = search::StackSearch;

    py::class_<ks>(m, "StackSearch", pydocs::DOC_StackSearch)
            .def(py::init<is&>())
            .def_property_readonly("num_images", &ks::num_images)
            .def_property_readonly("height", &ks::get_image_height)
            .def_property_readonly("width", &ks::get_image_width)
            .def_property_readonly("zeroed_times", &ks::get_zeroed_times)
            .def("search_all", &ks::search_all, pydocs::DOC_StackSearch_search)
            .def("evaluate_single_trajectory", &ks::evaluate_single_trajectory,
                 pydocs::DOC_StackSearch_evaluate_single_trajectory)
            .def("search_linear_trajectory", &ks::search_linear_trajectory,
                 pydocs::DOC_StackSearch_search_linear_trajectory)
            .def("set_min_obs", &ks::set_min_obs, pydocs::DOC_StackSearch_set_min_obs)
            .def("set_min_lh", &ks::set_min_lh, pydocs::DOC_StackSearch_set_min_lh)
            .def("set_results_per_pixel", &ks::set_results_per_pixel,
                 pydocs::DOC_StackSearch_set_results_per_pixel)
            .def("disable_gpu_sigmag_filter", &ks::disable_gpu_sigmag_filter,
                 pydocs::DOC_StackSearch_disable_gpu_sigmag_filter)
            .def("enable_gpu_sigmag_filter", &ks::enable_gpu_sigmag_filter,
                 pydocs::DOC_StackSearch_enable_gpu_sigmag_filter)
            .def("enable_gpu_encoding", &ks::enable_gpu_encoding, pydocs::DOC_StackSearch_enable_gpu_encoding)
            .def("set_start_bounds_x", &ks::set_start_bounds_x, pydocs::DOC_StackSearch_set_start_bounds_x)
            .def("set_start_bounds_y", &ks::set_start_bounds_y, pydocs::DOC_StackSearch_set_start_bounds_y)
            .def("get_num_images", &ks::num_images, pydocs::DOC_StackSearch_get_num_images)
            .def("get_image_width", &ks::get_image_width, pydocs::DOC_StackSearch_get_image_width)
            .def("get_image_height", &ks::get_image_height, pydocs::DOC_StackSearch_get_image_height)
            .def("get_all_psi_phi_curves", &ks::get_all_psi_phi_curves,
                 pydocs::DOC_StackSearch_get_all_psi_phi_curves)
            // For testings
            .def("prepare_psi_phi", &ks::prepare_psi_phi, pydocs::DOC_StackSearch_prepare_psi_phi)
            .def("clear_psi_phi", &ks::clear_psi_phi, pydocs::DOC_StackSearch_clear_psi_phi)
            .def("get_number_total_results", &ks::get_number_total_results,
                 pydocs::DOC_StackSearch_get_number_total_results)
            .def("get_results", &ks::get_results, pydocs::DOC_StackSearch_get_results)
            .def("get_all_results", &ks::get_all_results, pydocs::DOC_StackSearch_get_all_results)
            .def("set_results", &ks::set_results, pydocs::DOC_StackSearch_set_results)
            .def("clear_results", &ks::clear_results, pydocs::DOC_StackSearch_clear_results)
            .def("compute_max_results", &ks::compute_max_results,
                 pydocs::DOC_StackSearch_compute_max_results);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
