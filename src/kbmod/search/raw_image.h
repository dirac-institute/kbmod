#ifndef RAWIMAGEEIGEN_H_
#define RAWIMAGEEIGEN_H_

#include <vector>
#include <float.h>
#include <iostream>
#include <stdexcept>
#include <math.h>

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
    explicit RawImage(Image& img);
    explicit RawImage(unsigned w, unsigned h, float value = 0.0);

    RawImage(const RawImage& old) noexcept;  // Copy constructor
    RawImage(RawImage&& source) noexcept;    // Move constructor

    RawImage& operator=(const RawImage& source) noexcept;  // Copy assignment
    RawImage& operator=(RawImage&& source) noexcept;       // Move assignment

    // Basic getter functions for image data.
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    uint64_t get_npixels() const { return width * height; }
    const Image& get_image() const { return image; }
    Image& get_image() { return image; }
    void set_image(Image& other) { image = other; }

    inline bool contains(const Index& idx) const {
        return idx.i >= 0 && idx.i < height && idx.j >= 0 && idx.j < width;
    }

    inline bool contains(const Point& p) const { return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height; }

    inline float get_pixel(const Index& idx) const { return contains(idx) ? image(idx.i, idx.j) : NO_DATA; }

    inline void set_pixel(const Index& idx, float value) {
        if (!contains(idx)) throw std::runtime_error("Index out of bounds!");
        image(idx.i, idx.j) = value;
    }

    void set_all(float value);

    // Functions for determining and setting whether pixels have valid data.
    inline bool pixel_has_data(const Index& idx) const {
        return pixel_value_valid(get_pixel(idx)) ? true : false;
    }

    inline void mask_pixel(const Index& idx) {
        if (!contains(idx)) throw std::runtime_error("Index out of bounds!");
        image(idx.i, idx.j) = NO_DATA;
    }

    void replace_masked_values(float value = 0.0);

    // this will be a raw pointer to the underlying array
    // we use this to copy to GPU and nowhere else!
    float* data() { return image.data(); }

    // Check if two raw images are approximately equal. Counts invalid pixels
    // (NaNs) as equal if they appear in both images.
    bool l2_allclose(const RawImage& imgB, float atol) const;

    // Create a "stamp" image of a give radius (width=2*radius+1) about the
    // given point.
    // keep_no_data indicates whether to use the NO_DATA flag or replace with 0.0.
    RawImage create_stamp(const Point& p, const int radius, const bool keep_no_data) const;

    // Compute the min and max bounds of values in the image.
    std::array<float, 2> compute_bounds() const;

    // Convolve the image with a point spread function.
    void convolve(PSF& psf);
    void convolve_cpu(PSF& psf);

    // Masks out the array of the image where 'flags' is a bit vector of mask flags
    // to apply (use 0xFFFFFF to apply all flags).
    void apply_mask(int flags, const RawImage& mask);

    // The maximum value of the image and return the coordinates. The parameter
    // furthest_from_center indicates whether to break ties using the peak further
    // or closer to the center of the image.
    Index find_peak(bool furthest_from_center) const;

    // Find the basic image moments in order to test if stamps have a gaussian shape.
    // It computes the moments on the "normalized" image where the minimum
    // value has been shifted to zero and the sum of all elements is 1.0.
    // Elements with NO_DATA, NaN, etc. are treated as zero.
    ImageMoments find_central_moments() const;

    bool center_is_local_max(double flux_thresh, bool local_max) const;

    virtual ~RawImage(){};

private:
    unsigned width;
    unsigned height;
    Image image;
};

// Helper functions for creating composite images.
RawImage create_median_image(const std::vector<RawImage>& images);
RawImage create_summed_image(const std::vector<RawImage>& images);
RawImage create_mean_image(const std::vector<RawImage>& images);

} /* namespace search */

#endif /* RAWIMAGEEIGEN_H_ */
