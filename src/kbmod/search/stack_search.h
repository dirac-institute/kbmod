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

#include "logging.h"
#include "common.h"
#include "debug_timer.h"
#include "geom.h"
#include "image_stack.h"
#include "psf.h"
#include "psi_phi_array_ds.h"
#include "psi_phi_array_utils.h"
#include "pydocs/stack_search_docs.h"
#include "stamp_creator.h"
#include "trajectory_list.h"

namespace search {
using Point = indexing::Point;
using Image = search::Image;

class StackSearch {
public:
    StackSearch(ImageStack& imstack);
    int compute_max_results();
    int num_images() const { return stack.img_count(); }
    int get_image_width() const { return stack.get_width(); }
    int get_image_height() const { return stack.get_height(); }
    int get_image_npixels() const { return stack.get_npixels(); }
    const ImageStack& get_imagestack() const { return stack; }

    // Parameter setters used to control the searches.
    void set_min_obs(int new_value);
    void set_min_lh(float new_value);
    void enable_gpu_sigmag_filter(std::vector<float> percentiles, float sigmag_coeff, float min_lh);
    void enable_gpu_encoding(int num_bytes);
    void set_start_bounds_x(int x_min, int x_max);
    void set_start_bounds_y(int y_min, int y_max);
    void set_results_per_pixel(int new_value);

    // The primary search functions
    void evaluate_single_trajectory(Trajectory& trj);
    Trajectory search_linear_trajectory(int x, int y, float vx, float vy);
    void prepare_search(std::vector<Trajectory>& search_list, int min_observations);
    std::vector<Trajectory> search_single_batch();
    void search_batch();
    void search_all(std::vector<Trajectory>& search_list, int min_observations);
    void finish_search();

    // Gets the vector of result trajectories from the grid search.
    std::vector<Trajectory> get_results(int start, int end);

    // Getters for the Psi and Phi data.
    std::vector<float> get_psi_curves(Trajectory& t);
    std::vector<float> get_phi_curves(Trajectory& t);

    // Helper functions for computing Psi and Phi
    void prepare_psi_phi();
    void clear_psi_phi();

    // Helper functions for testing
    void set_results(const std::vector<Trajectory>& new_results);

    virtual ~StackSearch(){};

protected:
    std::vector<float> extract_psi_or_phi_curve(Trajectory& trj, bool extract_psi);

    // Core data and search parameters
    ImageStack stack;
    SearchParameters params;

    // Precomputed and cached search data
    bool psi_phi_generated;
    PsiPhiArray psi_phi_array;

    // Results from the grid search.
    TrajectoryList results;

    // Trajectories that are being searched.
    TrajectoryList gpu_search_list;

    // Logger for this object. Retrieved once this is used frequently.
    logging::Logger* rs_logger;
};

} /* namespace search */

#endif /* KBMODSEARCH_H_ */
