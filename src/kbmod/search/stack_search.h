#ifndef KBMODSEARCH_H_
#define KBMODSEARCH_H_

#include <parallel/algorithm>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>  // formatting log msgs
#include <chrono>
#include <stdexcept>
#include <float.h>
#include <omp.h>

#include "logging.h"
#include "common.h"
#include "cpu_search_algorithms.h"
#include "debug_timer.h"
#include "image_stack.h"
#include "psi_phi_array_ds.h"
#include "psi_phi_array_utils.h"
#include "pydocs/stack_search_docs.h"
#include "trajectory_list.h"

namespace search {
using Image = search::Image;

class StackSearch {
public:
    StackSearch(ImageStack& imstack);
    uint64_t compute_max_results();
    unsigned int num_images() const { return num_imgs; }
    unsigned int get_image_width() const { return width; }
    unsigned int get_image_height() const { return height; }
    std::vector<double>& get_zeroed_times() { return zeroed_times; }

    // Parameter setters used to control the searches.
    void set_default_parameters();
    void set_min_obs(int new_value);
    void set_min_lh(float new_value);
    void disable_gpu_sigmag_filter();
    void enable_gpu_sigmag_filter(std::vector<float> percentiles, float sigmag_coeff, float min_lh);
    void enable_gpu_encoding(int num_bytes);
    void set_start_bounds_x(int x_min, int x_max);
    void set_start_bounds_y(int y_min, int y_max);
    void set_results_per_pixel(int new_value);

    // The primary search functions
    void evaluate_single_trajectory(Trajectory& trj, bool use_kernel);
    Trajectory search_linear_trajectory(int x, int y, float vx, float vy, bool use_kernel);
    void search_all(std::vector<Trajectory>& search_list, bool on_gpu);

    // Gets the vector of result trajectories from the grid search.
    uint64_t get_number_total_results() { return results.get_size(); }
    std::vector<Trajectory> get_results(uint64_t start, uint64_t count);
    std::vector<Trajectory>& get_all_results();
    void clear_results();

    // Getters for the Psi and Phi data.
    Image get_all_psi_phi_curves(const std::vector<Trajectory>& trajectories);

    // Helper functions for computing Psi and Phi
    void prepare_psi_phi();
    void clear_psi_phi();

    // Helper functions for testing
    void set_results(const std::vector<Trajectory>& new_results);

    virtual ~StackSearch();

protected:
    // Core data and search parameters. Note the StackSearch does not own
    // the ImageStack and it must exist for the duration of the object's life.
    SearchParameters params;
    unsigned int height;
    unsigned int width;
    unsigned int num_imgs;
    std::vector<double> zeroed_times;

    ImageStack& stack;

    // Precomputed and cached search data
    bool psi_phi_generated;
    PsiPhiArray psi_phi_array;

    // Results from the grid search.
    TrajectoryList results;

    // Logger for this object. Retrieved once this is used frequently.
    logging::Logger* rs_logger;
};

} /* namespace search */

#endif /* KBMODSEARCH_H_ */
