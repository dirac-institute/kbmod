/*
 * LayeredImage.h
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 *
 *  LayeredImage stores an image from a single time with different layers of
 *  data, such as science pixels, variance pixels, and mask pixels.
 */

#ifndef LAYEREDIMAGE_H_
#define LAYEREDIMAGE_H_

#include <vector>
#include <fitsio.h>
#include <iostream>
#include <string>
#include <random>
#include <assert.h>
#include <stdexcept>
#include "raw_image.h"
#include "common.h"
#include "pydocs/layered_image_docs.h"


namespace search {

  class LayeredImage {
  public:
    explicit LayeredImage(std::string path, const PSF& psf);
    explicit LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk,
                          const PSF& psf);
    explicit LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                          const PSF& psf);
    explicit LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                          const PSF& psf, int seed);

    // Set an image specific point spread function.
    void set_psf(const PSF& psf);
    const PSF& get_psf() const { return psf; }

    // Basic getter functions for image data.
    std::string get_name() const { return filename; }
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    unsigned get_npixels() const { return width * height; }
    double get_obstime() const { return science.get_obstime(); }
    void set_obstime(double obstime) { science.set_obstime(obstime); }

    // Getter functions for the data in the individual layers.
    RawImage& get_science() { return science; }
    RawImage& get_mask() { return mask; }
    RawImage& get_variance() { return variance; }

    // Get pointers to the raw pixel arrays.
    float* getSDataRef() { return science.getDataRef(); }
    float* getVDataRef() { return variance.getDataRef(); }
    float* getMDataRef() { return mask.getDataRef(); }

    // Applies the mask functions to each of the science and variance layers.
    void apply_mask_flags(int flag, const std::vector<int>& exceptions);
    void applyGlobalMask(const RawImage& global_mask);
    void apply_mask_threshold(float thresh);
    void grow_mask(int steps);

    // Subtracts a template image from the science layer.
    void subtract_template(const RawImage& sub_template);

    // Saves the data in each later to a file.
    void save_layers(const std::string& path);

    // Setter functions for the individual layers.
    void set_science(RawImage& im);
    void set_mask(RawImage& im);
    void set_variance(RawImage& im);

    // Convolve with a given PSF or the default one.
    void convolve_psf();
    void convolve_given_psf(const PSF& psf);

    virtual ~LayeredImage(){};

    // Generate psi and phi images from the science and variance layers.
    RawImage generate_psi_image();
    RawImage generate_phi_image();

  private:
    void checkDims(RawImage& im);

    std::string filename;
    unsigned width;
    unsigned height;

    PSF psf;
    RawImage science;
    RawImage mask;
    RawImage variance;
  };

} /* namespace search */

#endif /* LAYEREDIMAGE_H_ */
