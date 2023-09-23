/*
 * raw_image.h
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 *
 * RawImage stores pixel level data for a single image.
 */

#ifndef RAWIMAGE_H_
#define RAWIMAGE_H_

#include <algorithm>
#include <array>
#include <vector>
#include <fitsio.h>
#include <float.h>
#include <iostream>
#include <string>
#include <assert.h>
#include <stdexcept>
#include "common.h"
#include "psf.h"
#include "pydocs/raw_image_docs.h"


namespace search {
  class RawImage {
  public:
    RawImage();
    RawImage(const RawImage& old);  // Copy constructor
    RawImage(RawImage&& source);    // Move constructor
    explicit RawImage(unsigned w, unsigned h);
    explicit RawImage(unsigned w, unsigned h, const std::vector<float>& pix);

#ifdef Py_PYTHON_H
    explicit RawImage(pybind11::array_t<float> arr);
    void set_array(pybind11::array_t<float>& arr);
#endif

    RawImage& operator=(const RawImage& source);  // Copy assignment
    RawImage& operator=(RawImage&& source);       // Move assignment

    // Basic getter functions for image data.
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    unsigned get_npixels() const { return width * height; }

    // Inline pixel functions.
    float get_pixel(int x, int y) const {
      return (x >= 0 && x < width && y >= 0 && y < height) ? pixels[y * width + x] : NO_DATA;
    }

    bool pixel_has_data(int x, int y) const {
      return (x >= 0 && x < width && y >= 0 && y < height) ? pixels[y * width + x] != NO_DATA : false;
    }

    void set_pixel(int x, int y, float value) {
      if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] = value;
    }
    const std::vector<float>& get_pixels() const { return pixels; }
    float* getDataRef() { return pixels.data(); }  // Get pointer to pixels

    // Get the interpolated brightness of a real values point
    // using the four neighboring pixels.
    float get_pixel_interp(float x, float y) const;

    // Check if two raw images are approximately equal.
    bool approx_equal(const RawImage& imgB, float atol) const;

    // Functions for locally storing the image time.
    double get_obstime() const { return obstime; }
    void set_obstime(double new_time) { obstime = new_time; }

    // Compute the min and max bounds of values in the image.
    std::array<float, 2> compute_bounds() const;

    // Masks out the pixels of the image where:
    //   flags a bit vector of mask flags to apply
    //       (use 0xFFFFFF to apply all flags)
    //   exceptions is a vector of pixel flags to ignore
    //   mask is an image of bit vector mask flags
    void apply_mask(int flags, const std::vector<int>& exceptions, const RawImage& mask);

    void set_all_pix(float value);
    void add_to_pixel(float fx, float fy, float value);
    void addPixelInterp(float x, float y, float value);
    std::vector<float> bilinearInterp(float x, float y) const;

    // Grow the area of masked pixels.
    void grow_mask(int steps);

    // Load the image data from a specific layer of a FITS file.
    // Overwrites the current image data.
    void load_from_file(const std::string& file_path, int layer_num);

    // Save the RawImage to a file (single layer) or append the layer to an existing file.
    void save_to_file(const std::string& filename);
    void append_layer_to_file(const std::string& filename);

    // Convolve the image with a point spread function.
    void convolve(PSF psf);
    void convolve_cpu(const PSF& psf);

    // Create a "stamp" image of a give radius (width=2*radius+1)
    // about the given point.
    // keep_no_data indicates whether to use the NO_DATA flag or replace with 0.0.
    RawImage create_stamp(float x, float y, int radius, bool interpolate, bool keep_no_data) const;

    // The maximum value of the image and return the coordinates. The parameter
    // furthest_from_center indicates whether to break ties using the peak further
    // or closer to the center of the image.
    PixelPos find_peak(bool furthest_from_center) const;

    // Find the basic image moments in order to test if stamps have a gaussian shape.
    // It computes the moments on the "normalized" image where the minimum
    // value has been shifted to zero and the sum of all elements is 1.0.
    // Elements with NO_DATA are treated as zero.
    ImageMoments find_central_moments() const;

    virtual ~RawImage(){};

  private:
    unsigned width;
    unsigned height;
    std::vector<float> pixels;
    double obstime;
  };

  // Helper functions for creating composite images.
  RawImage create_median_image(const std::vector<RawImage>& images);
  RawImage create_summed_image(const std::vector<RawImage>& images);
  RawImage create_mean_image(const std::vector<RawImage>& images);

} /* namespace search */

#endif /* RAWIMAGE_H_ */
