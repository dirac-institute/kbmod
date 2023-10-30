#include "layered_image.h"

namespace search {

  LayeredImage::LayeredImage(std::string path, const PSF& psf) : psf(psf) {
    int f_begin = path.find_last_of("/");
    int f_end = path.find_last_of(".fits") - 4;
    filename = path.substr(f_begin, f_end - f_begin);

    science = RawImage();
    science.from_fits(path, 1);
    width = science.get_width();
    height = science.get_height();

    mask = RawImage();
    mask.from_fits(path, 2);

    variance = RawImage();
    variance.from_fits(path, 3);

    if (width != variance.get_width() or height != variance.get_height())
      throw std::runtime_error("Science and Variance layers are not the same size.");
    if (width != mask.get_width() or height != mask.get_height())
      throw std::runtime_error("Science and Mask layers are not the same size.");
  }

  LayeredImage::LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk, const PSF& psf)
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

  LayeredImage::LayeredImage(std::string name, unsigned w, unsigned h, float noise_stdev, float pixel_variance,
                             double time, const PSF& psf)
    : LayeredImage(name, w, h, noise_stdev, pixel_variance, time, psf, -1) {}

  LayeredImage::LayeredImage(std::string name, unsigned w, unsigned h, float noise_stdev, float pixel_variance,
                             double time, const PSF& psf, int seed)
    : filename(name), psf(psf), width(w), height(h) {
    std::random_device r;
    std::default_random_engine generator(r());
    if (seed >= 0) {
      generator.seed(seed);
    }
    std::normal_distribution<float> distrib(0.0, noise_stdev);
    auto gaussian = [&](float) { return distrib(generator); };

    // Evaluate gaussian for each of HxW matrix, no input needed,
    // ergo "nullary" expr. We have to eval the Nullary to be able to give
    // an lvalue to the constructor.
    search::Image tmp = search::Image::NullaryExpr(height, width, gaussian);
    science = RawImage(tmp, time);
    mask = RawImage(width, height, 0.0);
    variance = RawImage(width, height, pixel_variance);
  }

  void LayeredImage::set_psf(const PSF& new_psf) { psf = new_psf; }

  void LayeredImage::grow_mask(int steps) {
    science.grow_mask(steps);
    variance.grow_mask(steps);
  }

  void LayeredImage::convolve_given_psf(const PSF& given_psf) {
    science.convolve(given_psf);

    // Square the PSF use that on the variance image.
    PSF psfsq = PSF(given_psf);  // Copy
    psfsq.square_psf();
    variance.convolve(psfsq);
  }

  void LayeredImage::convolve_psf() { convolve_given_psf(psf); }

  void LayeredImage::apply_mask_flags(int flags, const std::vector<int>& exceptions) {
    science.apply_mask(flags, exceptions, mask);
    variance.apply_mask(flags, exceptions, mask);
  }

  /* Mask all pixels that are not 0 in global mask */
  void LayeredImage::apply_global_mask(const RawImage& global_mask) {
    science.apply_mask(0xFFFFFF, {}, global_mask);
    variance.apply_mask(0xFFFFFF, {}, global_mask);
  }

  void LayeredImage::apply_mask_threshold(float thresh) {
    const int num_pixels = get_npixels();
    float* sci_pixels = science.data();
    float* var_pix = variance.data();
    for (int i = 0; i < num_pixels; ++i) {
      if (sci_pixels[i] > thresh) {
        sci_pixels[i] = NO_DATA;
        var_pix[i] = NO_DATA;
      }
    }
  }

  void LayeredImage::subtract_template(RawImage& sub_template) {
    assert(get_height() == sub_template.get_height() && get_width() == sub_template.get_width());
    const int num_pixels = get_npixels();

    float* sci_pixels = science.data();
    float* tem_pixels = sub_template.data();
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

    science.append_to_fits(path + filename + ".fits");
    mask.append_to_fits(path + filename + ".fits");
    variance.append_to_fits(path + filename + ".fits");
  }

  void LayeredImage::set_science(RawImage& im) {
    check_dims(im);
    science = im;
  }

  void LayeredImage::set_mask(RawImage& im) {
    check_dims(im);
    mask = im;
  }

  void LayeredImage::set_variance(RawImage& im) {
    check_dims(im);
    variance = im;
  }

  void LayeredImage::check_dims(RawImage& im) {
    if (im.get_width() != get_width()) throw std::runtime_error("Image width does not match");
    if (im.get_height() != get_height()) throw std::runtime_error("Image height does not match");
  }

  RawImage LayeredImage::generate_psi_image() {
    RawImage result(width, height);
    float* result_arr = result.data();
    float* sci_array = science.data();
    float* var_array = variance.data();

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
    float* result_arr = result.data();
    float* var_array = variance.data();

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
    psfsq.square_psf();
    result.convolve(psfsq);

    return result;
  }

#ifdef Py_PYTHON_H
  static void layered_image_bindings(py::module& m) {
    using li = search::LayeredImage;
    using ri = search::RawImage;
    using pf = search::PSF;

    py::class_<li>(m, "LayeredImage", pydocs::DOC_LayeredImage)
      .def(py::init<const std::string, pf&>())
      .def(py::init<const ri&, const ri&, const ri&, pf&>())
      .def(py::init<std::string, int, int, double, float, float, pf&>())
      .def(py::init<std::string, int, int, double, float, float, pf&, int>())
      .def("set_psf", &li::set_psf, pydocs::DOC_LayeredImage_set_psf)
      .def("get_psf", &li::get_psf, py::return_value_policy::reference_internal,
           pydocs::DOC_LayeredImage_get_psf)
      .def("apply_mask_flags", &li::apply_mask_flags, pydocs::DOC_LayeredImage_apply_mask_flags)
      .def("apply_mask_threshold", &li::apply_mask_threshold,
           pydocs::DOC_LayeredImage_apply_mask_threshold)
      .def("sub_template", &li::subtract_template, pydocs::DOC_LayeredImage_sub_template)
      .def("save_layers", &li::save_layers, pydocs::DOC_LayeredImage_save_layers)
      .def("get_science", &li::get_science, py::return_value_policy::reference_internal,
           pydocs::DOC_LayeredImage_get_science)
      .def("get_mask", &li::get_mask, py::return_value_policy::reference_internal,
           pydocs::DOC_LayeredImage_get_mask)
      .def("get_variance", &li::get_variance, py::return_value_policy::reference_internal,
           pydocs::DOC_LayeredImage_get_variance)
      .def("set_science", &li::set_science, pydocs::DOC_LayeredImage_set_science)
      .def("set_mask", &li::set_mask, pydocs::DOC_LayeredImage_set_mask)
      .def("set_variance", &li::set_variance, pydocs::DOC_LayeredImage_set_variance)
      .def("convolve_psf", &li::convolve_psf, pydocs::DOC_LayeredImage_convolve_psf)
      .def("convolve_given_psf", &li::convolve_given_psf, pydocs::DOC_LayeredImage_convolve_given_psf)
      .def("grow_mask", &li::grow_mask, pydocs::DOC_LayeredImage_grow_mask)
      .def("get_name", &li::get_name, pydocs::DOC_LayeredImage_get_name)
      .def("get_width", &li::get_width, pydocs::DOC_LayeredImage_get_width)
      .def("get_height", &li::get_height, pydocs::DOC_LayeredImage_get_height)
      .def("get_npixels", &li::get_npixels, pydocs::DOC_LayeredImage_get_npixels)
      .def("get_obstime", &li::get_obstime, pydocs::DOC_LayeredImage_get_obstime)
      .def("set_obstime", &li::set_obstime, pydocs::DOC_LayeredImage_set_obstime)
      .def("generate_psi_image", &li::generate_psi_image, pydocs::DOC_LayeredImage_generate_psi_image)
      .def("generate_phi_image", &li::generate_phi_image, pydocs::DOC_LayeredImage_generate_phi_image);
  }
#endif /* Py_PYTHON_H */

} /* namespace search */
