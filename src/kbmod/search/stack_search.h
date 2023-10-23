#ifndef KBMODSEARCH_H_
#define KBMODSEARCH_H_


#include <parallel/algorithm>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <assert.h>
#include <float.h>

#include "common.h"
#include "debug_timer.h"
#include "geom.h"
#include "image_stack.h"
#include "psf.h"
#include "pydocs/stack_search_docs.h"
#include "stamp_creator.h"

namespace search {
  using Point = indexing::Point;
  using Image = search::Image;

  class StackSearch {
  public:
    StackSearch(ImageStack& imstack);

    int num_images() const { return stack.img_count(); }
    int get_image_width() const { return stack.get_width(); }
    int get_image_height() const { return stack.get_height(); }
    int get_image_npixels() const { return stack.get_npixels(); }
    const ImageStack& get_imagestack() const { return stack; }

    void set_debug(bool d);

    // The primary search functions.
    void enable_gpu_sigmag_filter(std::vector<float> percentiles, float sigmag_coeff, float min_lh);
    void enable_gpu_encoding(int psi_num_bytes, int phi_num_bytes);

    void set_start_bounds_x(int x_min, int x_max);
    void set_start_bounds_y(int y_min, int y_max);

    void search(int a_steps, int v_steps, float min_angle, float max_angle, float min_velocity,
                float max_velocity, int min_observations);

    // Gets the vector of result trajectories.
    std::vector<Trajectory> get_results(int start, int end);

    // Get the predicted (pixel) positions for a given trajectory.
    Point get_trajectory_position(const Trajectory& t, int i) const;
    std::vector<Point> get_trajectory_positions(Trajectory& t) const;

    // Filters the results based on various parameters.
    void filter_results(int min_observations);
    void filter_results_lh(float min_lh);

    // Getters for the Psi and Phi data.
    std::vector<RawImage>& get_psi_images();
    std::vector<RawImage>& get_phi_images();
    std::vector<float> get_psi_curves(Trajectory& t);
    std::vector<float> get_phi_curves(Trajectory& t);

    // Save internal data products to a file.
    void save_psiphi(const std::string& path);

    // Helper functions for computing Psi and Phi.
    void prepare_psi_phi();

    // Helper functions for testing.
    void set_results(const std::vector<Trajectory>& new_results);

    virtual ~StackSearch(){};

protected:
    void save_images(const std::string& path);
    void sort_results();
    std::vector<float> create_curves(Trajectory t, const std::vector<RawImage>& imgs);

    // Fill an interleaved vector for the GPU functions.
    void fill_psi_phi(const std::vector<RawImage>& psi_imgs, const std::vector<RawImage>& phi_imgs,
                      std::vector<float>* psi_vect, std::vector<float>* phi_vect);

    // Set the parameter min/max/scale from the psi/phi/other images.
    std::vector<scaleParameters> compute_image_scaling(const std::vector<RawImage>& vect,
                                                       int encoding_bytes) const;

    // Creates list of trajectories to search.
    void create_search_list(int angle_steps, int velocity_steps, float min_ang, float max_ang, float min_vel,
                            float max_vel);


    bool psi_phi_generated;
    bool debug_info;
    ImageStack stack;
    std::vector<Trajectory> search_list;
    std::vector<RawImage> psi_images;
    std::vector<RawImage> phi_images;
    std::vector<Trajectory> results;

    // Parameters for the GPU search.
    SearchParameters params;
};

} /* namespace search */

#endif /* KBMODSEARCH_H_ */
