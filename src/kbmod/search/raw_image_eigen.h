#ifndef RAWIMAGEEIGEN_H_
#define RAWIMAGEEIGEN_H_

#include <vector>
#include <fitsio.h>
#include <float.h>
#include <iostream>
#include <stdexcept>
#include <assert.h>

#include <Eigen/Core>

#include "common.h"
#include "geom.h"
#include "psf.h"
#include "pydocs/raw_image_docs.h"


namespace search {
  using Index = indexing::Index;
  using Point = indexing::Point;
  using Rect = indexing::Rectangle;

  using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ImageI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ImageRef = Eigen::Ref<Image>;
  using ImageIRef = Eigen::Ref<Image>;


  class RawImageEigen {
  public:
    explicit RawImageEigen();
    explicit RawImageEigen(Image& img, double obs_time=-1.0d);
    explicit RawImageEigen(unsigned h, unsigned w, float value=0.0f, double obs_time=-1.0d);

    RawImageEigen(const RawImageEigen& old);  // Copy constructor
    RawImageEigen(RawImageEigen&& source);    // Move constructor

    RawImageEigen& operator=(const RawImageEigen& source);  // Copy assignment
    RawImageEigen& operator=(RawImageEigen&& source);       // Move assignment

    // Basic getter functions for image data.
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    unsigned get_npixels() const { return width * height; }
    const Image& get_image() const { return image; }
    Image& get_image() { return image; }
    void set_image(Image& other) { image = other; }


    inline bool contains(const Index& idx) const {
      return idx.i>=0 && idx.i<width && idx.j>=0 && idx.j<height;
    }

    inline bool contains(const Point& p) const {
      return p.x>=0 && p.y<width && p.y>=0 && p.y<height;
    }

    inline float get_pixel(const Index& idx) const {
      return contains(idx) ? image(idx.j, idx.i) : NO_DATA;
    }

    bool pixel_has_data(const Index& idx) const {
      return get_pixel(idx) != NO_DATA ? true : false;
    }

    void set_pixel(const Index& idx, float value) {
      // we should probably be letting Eigen freak out about setting an impossible
      // index instead of silently just nod doing it; but this is how it is
      if (contains(idx))
        image(idx.j, idx.i) = value;
    }

    // Functions for locally storing the image time.
    double get_obstime() const { return obstime; }
    void set_obstime(double new_time) { obstime = new_time; }

    // this will be a raw pointer to the underlying array
    // we use this to copy to GPU and nowhere else!
    float* data() { return image.data(); }
    void set_all(float value);

    // Check if two raw images are approximately equal.
    bool l2_allclose(const RawImageEigen& imgB, float atol) const;

    // Get the interpolated brightness of a real values point
    // using the four neighboring array.
    inline auto get_interp_neighbors_and_weights(const Point& p) const;
    float interpolate(const Point& p) const;

    // Create a "stamp" image of a give radius (width=2*radius+1) about the
    // given point.
    // keep_no_data indicates whether to use the NO_DATA flag or replace with 0.0.
    Image create_stamp(const Point& p,  const int radius,
                       const bool interpolate, const bool keep_no_data) const;

    // pixel modifiers
    void add(const Index& idx, const float value);
    void add(const Point& p, const float value);
    void interpolated_add(const Point& p, const float value);

    // Compute the min and max bounds of values in the image.
    std::array<float, 2> compute_bounds() const;

    // Convolve the image with a point spread function.
    void convolve(PSF psf);
    void convolve_cpu(PSF& psf);

    // Masks out the array of the image where:
    //   flags a bit vector of mask flags to apply
    //       (use 0xFFFFFF to apply all flags)
    //   exceptions is a vector of pixel flags to ignore
    //   mask is an image of bit vector mask flags
    void apply_mask(int flags, const std::vector<int>& exceptions, const RawImageEigen& mask);

    // Grow the area of masked array.
    void grow_mask(int steps);

    // The maximum value of the image and return the coordinates. The parameter
    // furthest_from_center indicates whether to break ties using the peak further
    // or closer to the center of the image.
    Index find_peak(bool furthest_from_center) const;

    // Find the basic image moments in order to test if stamps have a gaussian shape.
    // It computes the moments on the "normalized" image where the minimum
    // value has been shifted to zero and the sum of all elements is 1.0.
    // Elements with NO_DATA are treated as zero.
    ImageMoments find_central_moments() const;

    // Load the image data from a specific layer of a FITS file.
    // Overwrites the current image data.
    void load_fits(const std::string& file_path, int layer_num);

    // Save the RawImageEigen to a file (single layer) or append the layer to an existing file.
    void save_fits(const std::string& filename);
    void append_fits_extension(const std::string& filename);

    virtual ~RawImageEigen(){};

  private:
    unsigned width;
    unsigned height;
    double obstime;
    Image image;
  };

  // Helper functions for creating composite images.
  RawImageEigen create_median_image_eigen(const std::vector<RawImageEigen>& images);
  RawImageEigen create_median_image_eigen2(const std::vector<RawImageEigen>& images);
  RawImageEigen create_summed_image_eigen(const std::vector<RawImageEigen>& images);
  RawImageEigen create_mean_image_eigen(const std::vector<RawImageEigen>& images);

} /* namespace search */

#endif /* RAWIMAGEEIGEN_H_ */
