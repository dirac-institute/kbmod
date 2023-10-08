#ifndef RAWIMAGEEIGEN_H_
#define RAWIMAGEEIGEN_H_


#include <vector>
#include <fitsio.h>
#include <float.h>
#include <iostream>
#include <stdexcept>
#include <Eigen/Core>

#include "common.h"
#include "psf.h"
#include "pydocs/raw_image_docs.h"


namespace search {

  struct Index {
    unsigned i;
    unsigned j;

    Index(int x, int y)
      : i(x), j(y) {}

    Index(int x, int y)
      : i(x), j(y) {}

    Index(float x, float y)
      : i(floor(x)), j(floor(y)) {}

    Index(float x, float y, bool floor)
      : i(static_cast<unsigned>(x)), j(static_cast<int>(y)) {}

    std::array<Index, 8> neighbors(){
      return {{
        {i-1, j-1}, {i, j-1}, {i+1, j-1},
        {i-1, j}, /* this */ {i+1, j},
        {i-1, j+1}, {i, j+1}, {i+1, j+1}
        }};
    }

    std::array<unsigned, 4> centered_block(const unsigned size, const unsigned width, const unsigned height){
      unsigned d = size/2;
      unsigned left_x = ((i-d >= 0) && (i-d < width)) ? i-d : i;
      unsigned right_x = ((i+d >= 0) && (i+d < width)) ? i+d : width - i;
      unsigned top_y = ((j-d >= 0) && (j-d < height)) ? j-d : j;
      unsigned bot_y = ((j+d >= 0) && (j+d < height)) ? j+d : height - i;

      unsigned dx = right_x - left_x;
      unsigned dy = bot_y - top_y;
      return {left_x, dx, top_y, dy};
    }

    friend std::ostream& operator<<(std::ostream& os, const Index& rc);
  };

  std::ostream& operator<<(std::ostream& os, const Index& rc){
    os << "x: " << rc.i << " y: " << rc.j;
    return os;
  }



  struct Point{
    float x;
    float y;

    Point(float xd, float yd)
      : x(xd), y(yd) {}

    Index to_index(){
      return Index(x, y);
    }

    std::array<Point, 4> nearest_pixel_coords()  {
      return {{
        {floor(x-0.5f)+0.5f, floor(y+0.5f)+0.5f},
        {floor(x+0.5f)+0.5f, floor(y+0.5f)+0.5f},
        {floor(x-0.5f)+0.5f, floor(y-0.5f)+0.5f},
        {floor(x+0.5f)+0.5f, floor(y-0.5f)+0.5f}
        }};
    }

    std::array<Index, 4> nearest_pixel_idxs(){
      return {
        {x-0.5f, y+0.5f},
        {x+0.5f, y+0.5f},
        {x-0.5f, y-0.5f},
        {x+0.5f, y-0.5f}
      };
    };

    friend std::ostream& operator<<(std::ostream& os, const Point& rc);
  };

  std::ostream& operator<<(std::ostream& os, const Point& rc){
    os << "x: " << rc.x << " y: " << rc.y;
    return os;
  }


  using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ImageI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ImageRef = Eigen::Ref<Image>;
  using ImageIRef = Eigen::Ref<Image>;


  class RawImageEigen {
  public:
    RawImageEigen();
    explicit RawImageEigen(Image& img, double obs_time=-1.0d);
    explicit RawImageEigen(unsigned w, unsigned h, float value=0.0f, double obs_time=-1.0d);

    RawImageEigen(const RawImageEigen& old);  // Copy constructor
    RawImageEigen(RawImageEigen&& source);    // Move constructor

    RawImageEigen& operator=(const RawImageEigen& source);  // Copy assignment
    RawImageEigen& operator=(RawImageEigen&& source);       // Move assignment

    // Basic getter functions for image data.
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    unsigned get_npixels() const { return width * height; }

    // Inline pixel functions.
    bool contains(float x, float y) const{
      return x>=0 && x<=width && y>=0 && y<=height;
    }

    bool contains(int i, int j) const{
      return x>=0 && x<=width && y>=0 && y<=height;
    }

    bool contains(Index idx) const{
      return idx.i>=0 && idx.i<=width && idx.j>=0 && idx.j<=height;
    }

    // Functions for locally storing the image time.
    double get_obstime() const { return obstime; }
    void set_obstime(double new_time) { obstime = new_time; }

    // this will be a raw pointer to the underlying array
    // we use this to copy to GPU and nowhere else!
    float* data() { return image.data(); }
    void set_all(float value);

    // Check if two raw images are approximately equal.
    bool isclose(const RawImageEigen& imgB, float atol) const;

    // Get the interpolated brightness of a real values point
    // using the four neighboring array.
    auto get_interp_neighbors_and_weights(const float x, const float y) const;
    float interpolate(const float x, const float y) const;
    float interpolate2(const float x, const float y) const;

    // Create a "stamp" image of a give radius (width=2*radius+1) about the
    // given point.
    // keep_no_data indicates whether to use the NO_DATA flag or replace with 0.0.
    RawImageEigen create_stamp(const float x, float y,
                               const int radius, const bool interpolate,
                               const bool keep_no_data) const;

    // pixel modifiers
    void add(const float x, const float y, const float value);
    void add(const unsigned x, const unsigned y, const float value);
    void add(const Index p, const float value);
    void interpolated_add(const float x, const float y, const float value);

    // Compute the min and max bounds of values in the image.
    std::array<float, 2> compute_bounds() const;

    // Convolve the image with a point spread function.
    void convolve(PSF psf);
    void convolve_cpu(PSF& psf);

    void apply_bitmask(const ImageI& bitmask);
    void apply_mask1(const ImageI& mask, const int flags);
    // Masks out the array of the image where:
    //   flags a bit vector of mask flags to apply
    //       (use 0xFFFFFF to apply all flags)
    //   exceptions is a vector of pixel flags to ignore
    //   mask is an image of bit vector mask flags
    void apply_mask(int flags, const std::vector<int>& exceptions, const RawImageEigen& mask);

    // Grow the area of masked array.
    void grow_mask(unsigned steps);

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
    void load_from_file(const std::string& file_path, int layer_num);

    // Save the RawImageEigen to a file (single layer) or append the layer to an existing file.
    void save_to_file(const std::string& filename);
    void append_layer_to_file(const std::string& filename);

    virtual ~RawImageEigen(){};

    Image image;
  private:
    unsigned width;
    unsigned height;
    double obstime;
  };

  // Helper functions for creating composite images.
  RawImageEigen create_median_image_eigen(const std::vector<RawImageEigen>& images);
  RawImageEigen create_median_image_eigen2(const std::vector<RawImageEigen>& images);
  RawImageEigen create_summed_image_eigen(const std::vector<RawImageEigen>& images);
  RawImageEigen create_mean_image_eigen(const std::vector<RawImageEigen>& images);

} /* namespace search */

#endif /* RAWIMAGEEIGEN_H_ */
