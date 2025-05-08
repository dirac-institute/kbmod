#include "layered_image.h"

namespace search {

LayeredImage::LayeredImage(Image& sci, Image& var, Image& msk, Image& psf, double obs_time)
        : science(std::move(sci)),
          variance(std::move(var)),
          mask(std::move(msk)),
          psf(psf),
          obstime(obs_time) {
    width = science.cols();
    height = science.rows();

    // Check that sizes are compatible.
    assert_sizes_equal(variance.cols(), width, "variance layer width");
    assert_sizes_equal(variance.rows(), height, "variance layer height");
    assert_sizes_equal(mask.cols(), width, "mask layer width");
    assert_sizes_equal(mask.rows(), height, "mask layer height");
}

// Copy constructor
LayeredImage::LayeredImage(const LayeredImage& source) noexcept {
    width = source.width;
    height = source.height;
    obstime = source.obstime;
    science = source.science;
    mask = source.mask;
    variance = source.variance;
    psf = source.psf;
}

// Move constructor
LayeredImage::LayeredImage(LayeredImage&& source) noexcept
        : width(source.width), height(source.height), obstime(source.obstime) {
    science = std::move(source.science);
    mask = std::move(source.mask);
    variance = std::move(source.variance);
    psf = std::move(source.psf);
}

// Copy assignment
LayeredImage& LayeredImage::operator=(const LayeredImage& source) noexcept {
    width = source.width;
    height = source.height;
    obstime = source.obstime;
    science = source.science;
    mask = source.mask;
    variance = source.variance;
    psf = source.psf;
    return *this;
}

// Move assignment
LayeredImage& LayeredImage::operator=(LayeredImage&& source) noexcept {
    if (this != &source) {
        width = source.width;
        height = source.height;
        obstime = source.obstime;
        science = std::move(source.science);
        mask = std::move(source.mask);
        variance = std::move(source.variance);
        psf = std::move(source.psf);
    }
    return *this;
}

void LayeredImage::set_psf(const Image& new_psf) { psf = new_psf; }

void LayeredImage::convolve_given_psf(Image& given_psf) {
    science = std::move(convolve_image(science, given_psf));

    // Square the PSF use that on the variance image.
    Image psfsq = square_psf(given_psf);
    variance = std::move(convolve_image(variance, psfsq));
}

void LayeredImage::convolve_psf() { convolve_given_psf(psf); }

Image LayeredImage::square_psf(Image& given_psf) {
    // Make a copy of the PSF.
    Image psf_sq = given_psf;

    for (int r = 0; r < given_psf.rows(); ++r) {
        for (int c = 0; c < given_psf.cols(); ++c) {
            psf_sq(r, c) = given_psf(r, c) * given_psf(r, c);
        }
    }
    return psf_sq;
}

void LayeredImage::apply_mask(int flags) {
    for (unsigned int r = 0; r < height; ++r) {
        for (unsigned int c = 0; c < width; ++c) {
            int pix_flags = static_cast<int>(mask(r, c));
            if ((flags & pix_flags) != 0) {
                science(r, c) = NO_DATA;
                variance(r, c) = NO_DATA;
            }
        }  // for r
    }      // for c
}

Image LayeredImage::generate_psi_image() {
    Image result = Image::Zero(height, width);
    float* result_arr = result.data();
    float* sci_array = science.data();
    float* var_array = variance.data();

    // Set each of the result pixels.
    const uint64_t num_pixels = get_npixels();
    uint64_t no_data_count = 0;
    for (uint64_t p = 0; p < num_pixels; ++p) {
        float var_pix = var_array[p];
        if (pixel_value_valid(var_pix) && var_pix != 0.0 && pixel_value_valid(sci_array[p])) {
            result_arr[p] = sci_array[p] / var_pix;
        } else {
            result_arr[p] = NO_DATA;
            no_data_count += 1;
        }
    }
    // Convolve with the PSF.
    return convolve_image(result, psf);
}

Image LayeredImage::generate_phi_image() {
    Image result = Image::Zero(height, width);
    float* result_arr = result.data();
    float* var_array = variance.data();

    // Set each of the result pixels.
    const uint64_t num_pixels = get_npixels();
    uint64_t no_data_count = 0;
    for (uint64_t p = 0; p < num_pixels; ++p) {
        float var_pix = var_array[p];
        if (pixel_value_valid(var_pix) && var_pix != 0.0) {
            result_arr[p] = 1.0 / var_pix;
        } else {
            result_arr[p] = NO_DATA;
            no_data_count += 1;
        }
    }

    // Convolve with the PSF squared.
    Image psfsq = square_psf(psf);  // Copy
    return convolve_image(result, psfsq);
}

#ifdef Py_PYTHON_H
static void layered_image_bindings(py::module& m) {
    using li = search::LayeredImage;

    py::class_<li>(m, "LayeredImage", pydocs::DOC_LayeredImage)
            .def(py::init<search::Image&, search::Image&, search::Image&, search::Image&, double>(),
                 py::arg("sci").noconvert(true), py::arg("var").noconvert(true),
                 py::arg("msk").noconvert(true), py::arg("psf"), py::arg("obs_time"))
            .def_property_readonly("height", &li::get_height)
            .def_property_readonly("width", &li::get_width)
            .def_property_readonly("sci", &li::get_science_array, py::return_value_policy::reference_internal)
            .def_property_readonly("mask", &li::get_mask_array, py::return_value_policy::reference_internal)
            .def_property_readonly("var", &li::get_variance_array,
                                   py::return_value_policy::reference_internal)
            .def_property("time", &li::get_obstime, &li::set_obstime)
            .def_property("psf", &li::get_psf, &li::set_psf, py::return_value_policy::reference_internal)
            .def("set_psf", &li::set_psf, pydocs::DOC_LayeredImage_set_psf)
            .def("get_psf", &li::get_psf, py::return_value_policy::reference_internal,
                 pydocs::DOC_LayeredImage_get_psf)
            .def("apply_mask", &li::apply_mask, pydocs::DOC_LayeredImage_apply_mask)
            .def("convolve_psf", &li::convolve_psf, pydocs::DOC_LayeredImage_convolve_psf)
            .def("convolve_given_psf", &li::convolve_given_psf, pydocs::DOC_LayeredImage_convolve_given_psf)
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
