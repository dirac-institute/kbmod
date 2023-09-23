/*
 * LayeredImage.cpp
 *
 *  Created on: Jul 11, 2017
 *      Author: kbmod-usr
 */

#include "LayeredImage.h"

namespace search {

  LayeredImage::LayeredImage(std::string path, const PSF& psf) : psf(psf) {
    int f_begin = path.find_last_of("/");
    int f_end = path.find_last_of(".fits") - 4;
    filename = path.substr(f_begin, f_end - f_begin);

    science = RawImage();
    science.load_from_file(path, 1);
    width = science.get_width();
    height = science.get_height();

    mask = RawImage();
    mask.load_from_file(path, 2);

    variance = RawImage();
    variance.load_from_file(path, 3);

    if (width != variance.get_width() or height != variance.get_height())
      throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != mask.get_width() or height != mask.get_height())
      throw std::runtime_error("Science and Mask layers are not the same size.");
  }

LayeredImage::LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk,
                           const PSF& psf)
        : psf(psf) {
    // Get the dimensions of the science layer and check for consistency with
    // the other two layers.
    width = sci.get_width();
    height = sci.get_height();
    if (width != var.get_width() or height != var.get_height())
      throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != msk.get_width() or height != msk.get_height())
      throw std::runtime_error("Science and Mask layers are not the same size.");

    // Copy the image layers.
    science = sci;
    mask = msk;
    variance = var;
  }

  LayeredImage::LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                             const PSF& psf)
    : LayeredImage(name, w, h, noise_stdev, pixel_variance, time, psf, -1) {}

LayeredImage::LayeredImage(std::string name, int w, int h, float noise_stdev, float pixel_variance, double time,
                           const PSF& psf, int seed)
        : psf(psf) {
    filename = name;
    width = w;
    height = h;

    std::vector<float> raw_sci(width * height);
    std::random_device r;
    std::default_random_engine generator(r());
    if (seed >= 0) {
      generator.seed(seed);
    }
    std::normal_distribution<float> distrib(0.0, noise_stdev);
    for (float& p : raw_sci) p = distrib(generator);

    science = RawImage(w, h, raw_sci);
    science.set_obstime(time);

    mask = RawImage(w, h, std::vector<float>(w * h, 0.0));
    variance = RawImage(w, h, std::vector<float>(w * h, pixel_variance));
  }

  void LayeredImage::set_psf(const PSF& new_psf) {
    psf = new_psf;
  }

  void LayeredImage::grow_mask(int steps) {
    science.grow_mask(steps);
    variance.grow_mask(steps);
  }

  void LayeredImage::convolveGivenPSF(const PSF& given_psf) {
    science.convolve(given_psf);

    // Square the PSF use that on the variance image.
    PSF psfsq = PSF(given_psf);  // Copy
    psfsq.squarePSF();
    variance.convolve(psfsq);
  }

  void LayeredImage::convolvePSF() {
    convolveGivenPSF(psf);
  }

  void LayeredImage::apply_mask_flags(int flags, const std::vector<int>& exceptions) {
    science.apply_mask(flags, exceptions, mask);
    variance.apply_mask(flags, exceptions, mask);
  }

  /* Mask all pixels that are not 0 in global mask */
  void LayeredImage::applyGlobalMask(const RawImage& global_mask) {
    science.apply_mask(0xFFFFFF, {}, global_mask);
    variance.apply_mask(0xFFFFFF, {}, global_mask);
  }

  void LayeredImage::apply_mask_threshold(float thresh) {
    const int num_pixels = get_npixels();
    float* sci_pixels = science.getDataRef();
    float* var_pix = variance.getDataRef();
    for (int i = 0; i < num_pixels; ++i) {
      if (sci_pixels[i] > thresh) {
        sci_pixels[i] = NO_DATA;
        var_pix[i] = NO_DATA;
      }
    }
  }

  void LayeredImage::subtract_template(const RawImage& sub_template) {
    assert(get_height() == sub_template.getHeight() && get_width() == sub_template.get_width());
    const int num_pixels = get_npixels();

    float* sci_pixels = science.getDataRef();
    const std::vector<float>& tem_pixels = sub_template.get_pixels();
    for (unsigned i = 0; i < num_pixels; ++i) {
      if ((sci_pixels[i] != NO_DATA) && (tem_pixels[i] != NO_DATA)) {
        sci_pixels[i] -= tem_pixels[i];
      }
    }
  }

  void LayeredImage::save_layers(const std::string& path) {
    fitsfile* fptr;
    int status = 0;
    long naxes[2] = {0, 0};
    double obstime = science.get_obstime();

    fits_create_file(&fptr, (path + filename + ".fits").c_str(), &status);

    // If we are unable to create the file, check if it already exists
    // and, if so, delete it and retry the create.
    if (status == 105) {
      status = 0;
      fits_open_file(&fptr, (path + filename + ".fits").c_str(), READWRITE, &status);
      if (status == 0) {
        fits_delete_file(fptr, &status);
        fits_create_file(&fptr, (path + filename + ".fits").c_str(), &status);
      }
    }

    fits_create_img(fptr, SHORT_IMG, 0, naxes, &status);
    fits_update_key(fptr, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);

    science.append_layer_to_file(path + filename + ".fits");
    mask.append_layer_to_file(path + filename + ".fits");
    variance.append_layer_to_file(path + filename + ".fits");
  }

  void LayeredImage::set_science(RawImage& im) {
    checkDims(im);
    science = im;
  }

  void LayeredImage::set_mask(RawImage& im) {
    checkDims(im);
    mask = im;
  }

  void LayeredImage::set_variance(RawImage& im) {
    checkDims(im);
    variance = im;
  }

  void LayeredImage::checkDims(RawImage& im) {
    if (im.get_width() != get_width()) throw std::runtime_error("Image width does not match");
    if (im.get_height() != get_height()) throw std::runtime_error("Image height does not match");
  }

  RawImage LayeredImage::generate_psi_image() {
    RawImage result(width, height);
    float* result_arr = result.getDataRef();
    float* sci_array = getSDataRef();
    float* var_array = getVDataRef();

    // Set each of the result pixels.
    const int num_pixels = get_npixels();
    for (int p = 0; p < num_pixels; ++p) {
      float var_pix = var_array[p];
      if (var_pix != NO_DATA) {
        result_arr[p] = sci_array[p] / var_pix;
      } else {
        result_arr[p] = NO_DATA;
      }
    }

    // Convolve with the PSF.
    result.convolve(psf);

    return result;
  }

  RawImage LayeredImage::generate_phi_image() {
    RawImage result(width, height);
    float* result_arr = result.getDataRef();
    float* var_array = getVDataRef();

    // Set each of the result pixels.
    const int num_pixels = get_npixels();
    for (int p = 0; p < num_pixels; ++p) {
      float var_pix = var_array[p];
      if (var_pix != NO_DATA) {
        result_arr[p] = 1.0 / var_pix;
      } else {
        result_arr[p] = NO_DATA;
      }
    }

    // Convolve with the PSF squared.
    PSF psfsq = PSF(psf);  // Copy
    psfsq.squarePSF();
    result.convolve(psfsq);

    return result;
  }

} /* namespace search */
