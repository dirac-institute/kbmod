#ifndef LAYEREDIMAGE_H_
#define LAYEREDIMAGE_H_

#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <stdexcept>

#include "raw_image.h"
#include "common.h"
#include "pydocs/layered_image_docs.h"
#include "logging.h"
#include "image_utils_cpp.h"

namespace search {
class LayeredImage {
public:
    explicit LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk, const Image& psf,
                          double obs_time = -1.0);

    // Build a layered image from the underlying matrices, taking ownership of the image data.
    explicit LayeredImage(Image& sci, Image& var, Image& msk, Image& psf, double obs_time);

    LayeredImage(const LayeredImage& source) noexcept;  // Copy constructor
    LayeredImage(LayeredImage&& source) noexcept;       // Move constructor

    LayeredImage& operator=(const LayeredImage& source) noexcept;  // Copy assignment
    LayeredImage& operator=(LayeredImage&& source) noexcept;       // Move assignment

    // Set an image specific point spread function.
    void set_psf(const Image& psf);
    Image& get_psf() { return psf; }

    // Basic getter functions for image data.
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    uint64_t get_npixels() const { return width * height; }
    double get_obstime() const { return obstime; }
    void set_obstime(double new_obstime) { obstime = new_obstime; }

    // Getter functions for the data in the individual layers.
    RawImage& get_science() { return science; }
    RawImage& get_mask() { return mask; }
    RawImage& get_variance() { return variance; }

    // Getter functions for the data in the individual layers as Images.
    Image& get_science_array() { return science.get_image(); }
    Image& get_mask_array() { return mask.get_image(); }
    Image& get_variance_array() { return variance.get_image(); }

    // Masking functions.
    void mask_pixel(const Index& idx);
    void binarize_mask(int flags_to_keep);
    void apply_mask(int flags);

    // Setter functions for the individual layers.
    void set_science(RawImage& im);
    void set_mask(RawImage& im);
    void set_variance(RawImage& im);

    // Convolve with a given PSF or the default one.
    void convolve_psf();
    void convolve_given_psf(Image& psf);
    Image square_psf(Image& psf);

    virtual ~LayeredImage(){};

    // Generate psi and phi images from the science and variance layers.
    Image generate_psi_image();
    Image generate_phi_image();

private:
    unsigned width;
    unsigned height;
    double obstime;

    Image psf;
    RawImage science;
    RawImage mask;
    RawImage variance;
};

} /* namespace search */

#endif /* LAYEREDIMAGE_H_ */
