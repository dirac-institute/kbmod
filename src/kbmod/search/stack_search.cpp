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

StackSearch::StackSearch(std::vector<Image>& sci_imgs, std::vector<Image>& var_imgs,
                         std::vector<Image>& psf_kernels, std::vector<double>& zeroed_times, int num_bytes)
        : results(0), zeroed_times(zeroed_times) {
    // Get the logger for this module.
    rs_logger = logging::getLogger("kbmod.search.run_search");

    // Get the image size data.
    num_imgs = sci_imgs.size();
    if (num_imgs == 0) {
        throw std::runtime_error("No images in the to process.");
    }
    if (sci_imgs.size() != var_imgs.size()) {
        throw std::runtime_error("The number of science and variance images do not match. Science: " +
                                 std::to_string(sci_imgs.size()) + ", Variance: " +
                                 std::to_string(var_imgs.size()));
    }
    if (sci_imgs.size() != psf_kernels.size()) {
        throw std::runtime_error("The number of science and PSF kernel images do not match. Science: " +
                                 std::to_string(sci_imgs.size()) + ", PSF Kernels: " +
                                 std::to_string(psf_kernels.size()));
    }
    if (sci_imgs.size() != zeroed_times.size()) {
        throw std::runtime_error("The number of science images and zeroed times do not match. Science: " +
                                 std::to_string(sci_imgs.size()) + ", Zeroed Times: " +
                                 std::to_string(zeroed_times.size()));
    }
    width = sci_imgs[0].cols();
    height = sci_imgs[0].rows();

    // Set the parameters for the search.
    set_default_parameters(num_bytes);

    // Compute the psi/phi array.
    DebugTimer timer = DebugTimer("preparing Psi and Phi images", rs_logger);
    fill_psi_phi_array_from_image_arrays(psi_phi_array, num_bytes, sci_imgs, var_imgs, psf_kernels,
                                         zeroed_times);
    psi_phi_preloaded = false;
    timer.stop();
}

StackSearch::~StackSearch() {
    // Clear the memory allocated for psi and phi.
    if (psi_phi_array.on_gpu()) {
        psi_phi_array.clear_from_gpu();
    }
    psi_phi_array.clear();
}

// --------------------------------------------
// Configuration functions
// --------------------------------------------

void StackSearch::set_default_parameters(int num_bytes) {
    // Default The Thresholds.
    params.min_observations = 0;
    params.min_lh = 0.0;

    // Default filtering arguments.
    params.do_sigmag_filter = false;
    params.sgl_L = 0.25;
    params.sgl_H = 0.75;
    params.sigmag_coeff = -1.0;

    // Default the encoding parameters.
    if (num_bytes == 1 || num_bytes == 2) {
        params.encode_num_bytes = num_bytes;
    } else if (num_bytes == -1 || num_bytes == 4) {
        params.encode_num_bytes = -1;
    } else {
        throw std::runtime_error("Invalid encoding size. Must be -1, 1, 2 or 4. Got " + std::to_string(num_bytes));
    }

    // Default the results per pixel.
    params.results_per_pixel = 8;

    // Default pixel starting bounds.
    params.x_start_min = 0;
    params.x_start_max = width;
    params.y_start_min = 0;
    params.y_start_max = height;
}

void StackSearch::set_min_obs(int new_value) {
    if (new_value < 0) throw std::runtime_error("min_obs must be >= 0. Got " + std::to_string(new_value));
    if (new_value > num_imgs)
        throw std::runtime_error("min_obs cannot be greater than the number of images. min_obs = " +
                                 std::to_string(new_value) + ", num_imgs = " + std::to_string(num_imgs) + ".");

    params.min_observations = new_value;
}

void StackSearch::set_min_lh(float new_value) { params.min_lh = new_value; }

void StackSearch::set_results_per_pixel(int new_value) {
    if (new_value <= 0) throw std::runtime_error("Invalid results per pixel. Got " + std::to_string(new_value));
    params.results_per_pixel = new_value;
}

void StackSearch::enable_gpu_sigmag_filter(std::vector<float> percentiles, float sigmag_coeff, float min_lh) {
    if ((percentiles.size() != 2) || (percentiles[0] >= percentiles[1]) || (percentiles[0] <= 0.0) ||
        (percentiles[1] >= 1.0)) {
        throw std::runtime_error("Invalid percentiles for sigma G filtering. Got [" +
                                 std::to_string(percentiles[0]) + ", " +
                                 std::to_string(percentiles[1]) + "].");
    }
    if (sigmag_coeff <= 0.0) {
        throw std::runtime_error("Invalid coefficient for sigma G filtering. Got " +
                                 std::to_string(sigmag_coeff) + ".");
    }

    params.do_sigmag_filter = true;
    params.sgl_L = percentiles[0];
    params.sgl_H = percentiles[1];
    params.sigmag_coeff = sigmag_coeff;
    params.min_lh = min_lh;
}

void StackSearch::disable_gpu_sigmag_filter() { params.do_sigmag_filter = false; }

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

void StackSearch::preload_psi_phi_array() {
    if (!psi_phi_array.on_gpu()) {
        psi_phi_array.move_to_gpu();
        psi_phi_preloaded = true;
    }
}

void StackSearch::unload_psi_phi_array() {
    if (psi_phi_array.on_gpu()) {
        psi_phi_array.clear_from_gpu();
        psi_phi_preloaded = false;
    }
}


// --------------------------------------------
// Core search functions
// --------------------------------------------

void StackSearch::evaluate_single_trajectory(Trajectory& trj, bool use_kernel) {
    if (!use_kernel) {
        evaluate_trajectory_cpu(psi_phi_array, trj);
    } else {
        if (!has_gpu()) throw std::runtime_error("GPU is not available for kernel evaluation.");
        if (psi_phi_array.get_num_times() >= MAX_NUM_IMAGES) {
            throw std::runtime_error("Too many images to evaluate on GPU. Max = " +
                                     std::to_string(MAX_NUM_IMAGES));
        }
#ifdef HAVE_CUDA
        evaluateTrajectory(psi_phi_array.get_meta_data(), psi_phi_array.get_cpu_array_ptr(),
                           psi_phi_array.get_cpu_time_array_ptr(), params, &trj);
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
    results.reset_all();

    DebugTimer search_timer = DebugTimer("Running search", rs_logger);
    if (on_gpu) {
        if (!has_gpu()) throw std::runtime_error("GPU is not available for search.");

        // Moved the needed data to the GPU.
        rs_logger->info("Moving all data to GPU.");
        if (!psi_phi_preloaded) psi_phi_array.move_to_gpu();
        candidate_list.move_to_gpu();
        results.move_to_gpu();

        // Do the actual search on the GPU.
#ifdef HAVE_CUDA
        deviceSearchFilter(psi_phi_array, params, candidate_list, results);
#endif

        // Free up the GPU memory.  Keep the psi/phi array on the GPU if
        // it is preloaded.
        rs_logger->info("Clearing all data from GPU.");
        results.move_to_cpu();
        candidate_list.move_to_cpu();
        if (!psi_phi_preloaded) psi_phi_array.clear_from_gpu();
    } else {
        rs_logger->info("Running search on CPU.");
        search_cpu_only(psi_phi_array, params, candidate_list, results);
    }
    search_timer.stop();
    uint64_t num_results = results.get_size();
    rs_logger->debug("Core search returned " + std::to_string(num_results) + " results.\n");

    // Perform initial LH and obs_count filtering.
    DebugTimer filter_timer = DebugTimer("Filtering results by LH and min_obs", rs_logger);
    results.filter_by_likelihood(params.min_lh);
    results.filter_by_obs_count(params.min_observations);
    uint64_t new_num_results = results.get_size();
    rs_logger->debug("After filtering by LH and min_obs " + std::to_string(new_num_results) + " results (" +
                     std::to_string(num_results - new_num_results) + " removed).\n");
    filter_timer.stop();

    // Sort the results by decreasing likelihood.
    DebugTimer sort_timer = DebugTimer("Sorting results", rs_logger);
    results.sort_by_likelihood();
    sort_timer.stop();

    // Check that all trajectories left are valid.
    results.assert_valid();

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
    using iv = std::vector<search::Image>;
    using dv = std::vector<double>;
    using tj = search::Trajectory;
    using ks = search::StackSearch;

    py::class_<ks>(m, "StackSearch", pydocs::DOC_StackSearch)
            .def(py::init<iv&, iv&, iv&, dv&, int>(), py::arg("sci_imgs"), py::arg("var_imgs"),
                 py::arg("psf_kernels"), py::arg("zeroed_times"), py::arg("num_bytes") = -1)
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
            .def("set_start_bounds_x", &ks::set_start_bounds_x, pydocs::DOC_StackSearch_set_start_bounds_x)
            .def("set_start_bounds_y", &ks::set_start_bounds_y, pydocs::DOC_StackSearch_set_start_bounds_y)
            .def("get_num_images", &ks::num_images, pydocs::DOC_StackSearch_get_num_images)
            .def("get_image_width", &ks::get_image_width, pydocs::DOC_StackSearch_get_image_width)
            .def("get_image_height", &ks::get_image_height, pydocs::DOC_StackSearch_get_image_height)
            .def("get_all_psi_phi_curves", &ks::get_all_psi_phi_curves,
                 pydocs::DOC_StackSearch_get_all_psi_phi_curves)
            .def("preload_psi_phi_array", &ks::preload_psi_phi_array,
                 pydocs::DOC_StackSearch_preload_psi_phi_array)
            .def("unload_psi_phi_array", &ks::unload_psi_phi_array,
                 pydocs::DOC_StackSearch_unload_psi_phi_array)
            .def("psi_phi_array_on_gpu", &ks::psi_phi_array_on_gpu,
                 pydocs::DOC_StackSearch_psi_phi_array_on_gpu)
            // For testings
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
