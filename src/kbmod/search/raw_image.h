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

using Image = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageRef = Eigen::Ref<Image>;
using ImageIRef = Eigen::Ref<Image>;

class RawImage {
public:
    explicit RawImage();
    explicit RawImage(Image& img, double obs_time = -1.0);
    explicit RawImage(unsigned h, unsigned w, float value = 0.0, double obs_time = -1.0);

    RawImage(const RawImage& old);  // Copy constructor
    RawImage(RawImage&& source);    // Move constructor

    RawImage& operator=(const RawImage& source);  // Copy assignment
    RawImage& operator=(RawImage&& source);       // Move assignment

    // Basic getter functions for image data.
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    unsigned get_npixels() const { return width * height; }
    const Image& get_image() const { return image; }
    Image& get_image() { return image; }
    void set_image(Image& other) { image = other; }

    inline bool contains(const Index& idx) const {
        return idx.i >= 0 && idx.i < width && idx.j >= 0 && idx.j < height;
    }

    inline bool contains(const Point& p) const { return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height; }

    inline float get_pixel(const Index& idx) const { return contains(idx) ? image(idx.j, idx.i) : NO_DATA; }

    bool pixel_has_data(const Index& idx) const { return get_pixel(idx) != NO_DATA ? true : false; }

    void set_pixel(const Index& idx, float value) {
        // we should probably be letting Eigen freak out about setting an impossible
        // index instead of silently just nod doing it; but this is how it is
        if (contains(idx)) image(idx.j, idx.i) = value;
    }

    // Functions for locally storing the image time.
    double get_obstime() const { return obstime; }
    void set_obstime(double new_time) { obstime = new_time; }

    // this will be a raw pointer to the underlying array
    // we use this to copy to GPU and nowhere else!
    float* data() { return image.data(); }
    void set_all(float value);

    // Check if two raw images are approximately equal.
    bool l2_allclose(const RawImage& imgB, float atol) const;

    // Get the interpolated brightness of a real values point
    // using the four neighboring array.
    inline auto get_interp_neighbors_and_weights(const Point& p) const;
    float interpolate(const Point& p) const;

    // Create a "stamp" image of a give radius (width=2*radius+1) about the
    // given point.
    // keep_no_data indicates whether to use the NO_DATA flag or replace with 0.0.
    RawImage create_stamp(const Point& p, const int radius, const bool interpolate,
                          const bool keep_no_data) const;

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
    void apply_mask(int flags, const std::vector<int>& exceptions, const RawImage& mask);

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
    void from_fits(const std::string& file_path, int layer_num);

    // Save the RawImage to a file (single layer) or append the layer to an existing file.
    void to_fits(const std::string& filename);
    void append_to_fits(const std::string& filename);

    virtual ~RawImage(){};

private:
    void load_time_from_file(fitsfile* fptr);
    unsigned width;
    unsigned height;
    double obstime;
    Image image;
};

// Helper functions for creating composite images.
RawImage create_median_image(const std::vector<RawImage>& images);
RawImage create_summed_image(const std::vector<RawImage>& images);
RawImage create_mean_image(const std::vector<RawImage>& images);

} /* namespace search */

#endif /* RAWIMAGEEIGEN_H_ */
