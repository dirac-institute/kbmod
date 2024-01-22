#include "stack_search.h"

namespace search {
#ifdef HAVE_CUDA
extern "C" void deviceSearchFilter(PsiPhiArray& psi_phi_data, float* image_times, SearchParameters params,
                                   int num_trajectories, Trajectory* trj_to_search, int num_results,
                                   Trajectory* best_results);
#endif

StackSearch::StackSearch(ImageStack& imstack) : stack(imstack) {
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

void StackSearch::set_debug(bool d) {
    debug_info = d;
    params.debug = d;
}

void StackSearch::enable_gpu_sigmag_filter(std::vector<float> percentiles, float sigmag_coeff, float min_lh) {
    params.do_sigmag_filter = true;
    params.sgl_L = percentiles[0];
    params.sgl_H = percentiles[1];
    params.sigmag_coeff = sigmag_coeff;
    params.min_lh = min_lh;
}

void StackSearch::enable_gpu_encoding(int encode_num_bytes) {
    // Make sure the encoding is one of the supported options.
    // Otherwise use default float (aka no encoding).
    if (encode_num_bytes == 1 || encode_num_bytes == 2) {
        params.encode_num_bytes = encode_num_bytes;
    } else {
        params.encode_num_bytes = -1;
    }
}

void StackSearch::set_start_bounds_x(int x_min, int x_max) {
    params.x_start_min = x_min;
    params.x_start_max = x_max;
}

void StackSearch::set_start_bounds_y(int y_min, int y_max) {
    params.y_start_min = y_min;
    params.y_start_max = y_max;
}

void StackSearch::search(int ang_steps, int vel_steps, float min_ang, float max_ang, float min_vel,
                         float mavx, int min_observations) {
    DebugTimer core_timer = DebugTimer("Running core search", debug_info);
    create_search_list(ang_steps, vel_steps, min_ang, max_ang, min_vel, mavx);

    // Create a data stucture for the per-image data.
    std::vector<float> image_times = stack.build_zeroed_times();

    DebugTimer psi_phi_timer = DebugTimer("Creating psi/phi buffers", debug_info);
    prepare_psi_phi();
    PsiPhiArray psi_phi_data;
    fill_psi_phi_array(psi_phi_data, params.encode_num_bytes, psi_images, phi_images, debug_info);
    psi_phi_timer.stop();

    // Allocate a vector for the results.
    int num_search_pixels =
            ((params.x_start_max - params.x_start_min) * (params.y_start_max - params.y_start_min));
    int max_results = num_search_pixels * RESULTS_PER_PIXEL;
    if (debug_info) {
        std::cout << "Searching X=[" << params.x_start_min << ", " << params.x_start_max << "]"
                  << " Y=[" << params.y_start_min << ", " << params.y_start_max << "]\n";
        std::cout << "Allocating space for " << max_results << " results.\n";
    }
    results = std::vector<Trajectory>(max_results);
    if (debug_info) std::cout << search_list.size() << " trajectories... \n" << std::flush;

    // Set the minimum number of observations.
    params.min_observations = min_observations;

    // Do the actual search on the GPU.
    DebugTimer search_timer = DebugTimer("Running search", debug_info);
#ifdef HAVE_CUDA
    deviceSearchFilter(psi_phi_data, image_times.data(), params, search_list.size(), search_list.data(),
                       max_results, results.data());
#else
    throw std::runtime_error("Non-GPU search is not implemented.");
#endif
    search_timer.stop();

    DebugTimer sort_timer = DebugTimer("Sorting results", debug_info);
    sort_results();
    sort_timer.stop();
    core_timer.stop();
}

void StackSearch::prepare_psi_phi() {
    if (!psi_phi_generated) {
        DebugTimer timer = DebugTimer("Preparing Psi and Phi images", debug_info);
        psi_images.clear();
        phi_images.clear();

        // Compute Phi and Psi from convolved images
        // while leaving masked pixels alone
        // Reinsert 0s for NO_DATA?
        const int num_images = stack.img_count();
        for (int i = 0; i < num_images; ++i) {
            LayeredImage& img = stack.get_single_image(i);
            psi_images.push_back(img.generate_psi_image());
            phi_images.push_back(img.generate_phi_image());
        }

        psi_phi_generated = true;
        timer.stop();
    }
}

void StackSearch::create_search_list(int angle_steps, int velocity_steps, float min_ang, float max_ang,
                                     float min_vel, float mavx) {
    DebugTimer timer = DebugTimer("Creating search candidate list", debug_info);

    std::vector<float> angles(angle_steps);
    float ang_stepsize = (max_ang - min_ang) / float(angle_steps);
    for (int i = 0; i < angle_steps; ++i) {
        angles[i] = min_ang + float(i) * ang_stepsize;
    }

    std::vector<float> velocities(velocity_steps);
    float vel_stepsize = (mavx - min_vel) / float(velocity_steps);
    for (int i = 0; i < velocity_steps; ++i) {
        velocities[i] = min_vel + float(i) * vel_stepsize;
    }

    int trajCount = angle_steps * velocity_steps;
    search_list = std::vector<Trajectory>(trajCount);
    for (int a = 0; a < angle_steps; ++a) {
        for (int v = 0; v < velocity_steps; ++v) {
            search_list[a * velocity_steps + v].vx = cos(angles[a]) * velocities[v];
            search_list[a * velocity_steps + v].vy = sin(angles[a]) * velocities[v];
        }
    }
    timer.stop();
}

Point StackSearch::get_trajectory_position(const Trajectory& t, int i) const {
    float time = stack.get_zeroed_time(i);
    return {t.x + time * t.vx, t.y + time * t.vy};
}

std::vector<Point> StackSearch::get_trajectory_positions(Trajectory& t) const {
    std::vector<Point> results;
    int num_times = stack.img_count();
    for (int i = 0; i < num_times; ++i) {
        Point pos = get_trajectory_position(t, i);
        results.push_back(pos);
    }
    return results;
}

std::vector<float> StackSearch::create_curves(Trajectory t, const std::vector<RawImage>& imgs) {
    /*Create a lightcurve from an image along a trajectory
     *
     *  INPUT-
     *    Trajectory t - The trajectory along which to compute the lightcurve
     *    std::vector<RawImage*> imgs - The image from which to compute the
     *      trajectory. Most likely a psiImage or a phiImage.
     *  Output-
     *    std::vector<float> lightcurve - The computed trajectory
     */

    int img_size = imgs.size();
    std::vector<float> lightcurve;
    lightcurve.reserve(img_size);
    std::vector<float> times = stack.build_zeroed_times();
    for (int i = 0; i < img_size; ++i) {
        /* Do not use get_pixel_interp(), because results from create_curves must
         * be able to recover the same likelihoods as the ones reported by the
         * gpu search.*/
        Point p({t.x + times[i] * t.vx + 0.5f, t.y + times[i] * t.vy + 0.5f});
        float pix_val = imgs[i].get_pixel(p.to_index());
        if (pix_val == NO_DATA) pix_val = 0.0;
        lightcurve.push_back(pix_val);
    }
    return lightcurve;
}

std::vector<float> StackSearch::get_psi_curves(Trajectory& t) {
    /*Generate a psi lightcurve for further analysis
     *  INPUT-
     *    Trajectory& t - The trajectory along which to find the lightcurve
     *  OUTPUT-
     *    std::vector<float> - A vector of the lightcurve values
     */
    prepare_psi_phi();
    return create_curves(t, psi_images);
}

std::vector<float> StackSearch::get_phi_curves(Trajectory& t) {
    /*Generate a phi lightcurve for further analysis
     *  INPUT-
     *    Trajectory& t - The trajectory along which to find the lightcurve
     *  OUTPUT-
     *    std::vector<float> - A vector of the lightcurve values
     */
    prepare_psi_phi();
    return create_curves(t, phi_images);
}

std::vector<RawImage>& StackSearch::get_psi_images() { return psi_images; }

std::vector<RawImage>& StackSearch::get_phi_images() { return phi_images; }

void StackSearch::sort_results() {
    __gnu_parallel::sort(results.begin(), results.end(),
                         [](Trajectory a, Trajectory b) { return b.lh < a.lh; });
}

void StackSearch::filter_results(int min_observations) {
    results.erase(std::remove_if(results.begin(), results.end(),
                                 std::bind([](Trajectory t, int cutoff) { return t.obs_count < cutoff; },
                                           std::placeholders::_1, min_observations)),
                  results.end());
}

void StackSearch::filter_results_lh(float min_lh) {
    results.erase(std::remove_if(results.begin(), results.end(),
                                 std::bind([](Trajectory t, float cutoff) { return t.lh < cutoff; },
                                           std::placeholders::_1, min_lh)),
                  results.end());
}

std::vector<Trajectory> StackSearch::get_results(int start, int count) {
    if (start + count >= results.size()) {
        count = results.size() - start;
    }
    if (start < 0) throw std::runtime_error("start must be 0 or greater");
    return std::vector<Trajectory>(results.begin() + start, results.begin() + start + count);
}

// This function is used only for testing by injecting known result trajectories.
void StackSearch::set_results(const std::vector<Trajectory>& new_results) { results = new_results; }

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
            .def("enable_gpu_sigmag_filter", &ks::enable_gpu_sigmag_filter,
                 pydocs::DOC_StackSearch_enable_gpu_sigmag_filter)
            .def("enable_gpu_encoding", &ks::enable_gpu_encoding, pydocs::DOC_StackSearch_enable_gpu_encoding)
            .def("set_start_bounds_x", &ks::set_start_bounds_x, pydocs::DOC_StackSearch_set_start_bounds_x)
            .def("set_start_bounds_y", &ks::set_start_bounds_y, pydocs::DOC_StackSearch_set_start_bounds_y)
            .def("set_debug", &ks::set_debug, pydocs::DOC_StackSearch_set_debug)
            .def("filter_min_obs", &ks::filter_results, pydocs::DOC_StackSearch_filter_min_obs)
            .def("get_num_images", &ks::num_images, pydocs::DOC_StackSearch_get_num_images)
            .def("get_image_width", &ks::get_image_width, pydocs::DOC_StackSearch_get_image_width)
            .def("get_image_height", &ks::get_image_height, pydocs::DOC_StackSearch_get_image_height)
            .def("get_image_npixels", &ks::get_image_npixels, pydocs::DOC_StackSearch_get_image_npixels)
            .def("get_imagestack", &ks::get_imagestack, py::return_value_policy::reference_internal,
                 pydocs::DOC_StackSearch_get_imagestack)
            // For testings
            .def("get_trajectory_position", &ks::get_trajectory_position,
                 pydocs::DOC_StackSearch_get_trajectory_position)
            .def("get_psi_curves", (std::vector<float>(ks::*)(tj&)) & ks::get_psi_curves,
                 pydocs::DOC_StackSearch_get_psi_curves)
            .def("get_phi_curves", (std::vector<float>(ks::*)(tj&)) & ks::get_phi_curves,
                 pydocs::DOC_StackSearch_get_phi_curves)
            .def("prepare_psi_phi", &ks::prepare_psi_phi, pydocs::DOC_StackSearch_prepare_psi_phi)
            .def("get_psi_images", &ks::get_psi_images, pydocs::DOC_StackSearch_get_psi_images)
            .def("get_phi_images", &ks::get_phi_images, pydocs::DOC_StackSearch_get_phi_images)
            .def("get_results", &ks::get_results, pydocs::DOC_StackSearch_get_results)
            .def("set_results", &ks::set_results, pydocs::DOC_StackSearch_set_results);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
