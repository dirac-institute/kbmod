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
    explicit LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk, const PSF& psf);
    explicit LayeredImage(std::string name, unsigned width, unsigned height, float noise_stdev,
                          float pixel_variance, double time, const PSF& psf);
    explicit LayeredImage(std::string name, unsigned width, unsigned height, float noise_stdev,
                          float pixel_variance, double time, const PSF& psf, int seed);

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

    // Getter functions for the pixels of the science and variance layers that check
    // the mask layer for any set bits.
    inline float get_science_pixel(const Index& idx) const {
        // The get_pixel() functions perform the bounds checking and will return NO_DATA for out of bounds.
        return mask.get_pixel(idx) == 0 ? science.get_pixel(idx) : NO_DATA;
    }

    inline float get_variance_pixel(const Index& idx) const {
        // The get_pixel() functions perform the bounds checking and will return NO_DATA for out of bounds.
        return mask.get_pixel(idx) == 0 ? variance.get_pixel(idx) : NO_DATA;
    }

    inline bool contains(const Index& idx) const {
        return idx.i >= 0 && idx.i < height && idx.j >= 0 && idx.j < width;
    }

    // Masking functions.
    void binarize_mask(int flags_to_keep);
    void union_masks(RawImage& new_mask);
    void union_threshold_masking(float thresh);
    void grow_mask(int steps);
    void apply_mask(int flags);

    // Subtracts a template image from the science layer.
    void subtract_template(RawImage& sub_template);

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
    void check_dims(RawImage& im);

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
