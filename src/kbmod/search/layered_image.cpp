#include "layered_image.h"

namespace search {

LayeredImage::LayeredImage(const RawImage& sci, const RawImage& var, const RawImage& msk, const PSF& psf,
                           double obs_time)
        : obstime(obs_time), psf(psf) {
    // Get the dimensions of the science layer and check for consistency with
    // the other two layers.
    width = sci.get_width();
    height = sci.get_height();
    assert_sizes_equal(var.get_width(), width, "variance layer width");
    assert_sizes_equal(var.get_height(), height, "variance layer height");
    assert_sizes_equal(msk.get_width(), width, "mask layer width");
    assert_sizes_equal(msk.get_height(), height, "mask layer height");

    // Copy the image layers.
    science = sci;
    mask = msk;
    variance = var;
}

LayeredImage::LayeredImage(Image& sci, Image& var, Image& msk, PSF& psf, double obs_time)
        : science(sci), variance(var), mask(msk), psf(psf) {
    width = science.get_width();
    height = science.get_height();
    obstime = obs_time;
    assert_sizes_equal(variance.get_width(), width, "variance layer width");
    assert_sizes_equal(variance.get_height(), height, "variance layer height");
    assert_sizes_equal(mask.get_width(), width, "mask layer width");
    assert_sizes_equal(mask.get_height(), height, "mask layer height");
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
        : width(source.width),
          height(source.height),
          obstime(source.obstime),
          science(std::move(source.science)),
          mask(std::move(source.mask)),
          variance(std::move(source.variance)),
          psf(std::move(source.psf)) {}

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

void LayeredImage::set_psf(const PSF& new_psf) { psf = new_psf; }

void LayeredImage::convolve_given_psf(PSF& given_psf) {
    logging::getLogger("kbmod.search.layered_image")->debug("Convolving with " + given_psf.stats_string());
    science.convolve(given_psf);

    // Square the PSF use that on the variance image.
    PSF psfsq = PSF(given_psf);  // Copy
    psfsq.square_psf();
    variance.convolve(psfsq);
}

void LayeredImage::convolve_psf() { convolve_given_psf(psf); }

void LayeredImage::mask_pixel(const Index& idx) {
    science.mask_pixel(idx);
    variance.mask_pixel(idx);
    mask.set_pixel(idx, 1);
}

void LayeredImage::binarize_mask(int flags_to_use) {
    logging::getLogger("kbmod.search.layered_image")
            ->debug("Converting mask to binary using " + std::to_string(flags_to_use));

    const uint64_t num_pixels = get_npixels();
    float* mask_pixels = mask.data();

    for (uint64_t i = 0; i < num_pixels; ++i) {
        int current_flags = static_cast<int>(mask_pixels[i]);

        // Use a bitwise AND to only keep flags that are set in the current pixel
        // and in the flags_to_use bitmask.
        mask_pixels[i] = (flags_to_use & current_flags) > 0 ? 1 : 0;
    }
}

void LayeredImage::apply_mask(int flags) {
    science.apply_mask(flags, mask);
    variance.apply_mask(flags, mask);
}

void LayeredImage::set_science(RawImage& im) {
    assert_sizes_equal(im.get_width(), width, "science layer width");
    assert_sizes_equal(im.get_height(), height, "science layer height");
    science = im;
}

void LayeredImage::set_mask(RawImage& im) {
    assert_sizes_equal(im.get_width(), width, "mask layer width");
    assert_sizes_equal(im.get_height(), height, "mask layer height");
    mask = im;
}

void LayeredImage::set_variance(RawImage& im) {
    assert_sizes_equal(im.get_width(), width, "variance layer width");
    assert_sizes_equal(im.get_height(), height, "variance layer height");
    variance = im;
}

RawImage LayeredImage::generate_psi_image() {
    RawImage result(width, height);
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
    result.convolve(psf);

    logging::getLogger("kbmod.search.layered_image")
            ->debug("Generated psi image. " + std::to_string(no_data_count) + " of " +
                    std::to_string(num_pixels) + " had no data.");

    return result;
}

RawImage LayeredImage::generate_phi_image() {
    RawImage result(width, height);
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
    PSF psfsq = PSF(psf);  // Copy
    psfsq.square_psf();
    result.convolve(psfsq);

    logging::getLogger("kbmod.search.layered_image")
            ->debug("Generated phi image. " + std::to_string(no_data_count) + " of " +
                    std::to_string(num_pixels) + " had no data.");

    return result;
}

#ifdef Py_PYTHON_H
static void layered_image_bindings(py::module& m) {
    using li = search::LayeredImage;
    using ri = search::RawImage;
    using pf = search::PSF;

    py::class_<li>(m, "LayeredImage", pydocs::DOC_LayeredImage)
            .def(py::init<const ri&, const ri&, const ri&, const pf&, double>(), py::arg("sci"),
                 py::arg("var"), py::arg("msk"), py::arg("psf"), py::arg("obs_time") = -1.0)
            .def(py::init<search::Image&, search::Image&, search::Image&, pf&, double>(),
                 py::arg("sci").noconvert(true), py::arg("var").noconvert(true),
                 py::arg("msk").noconvert(true), py::arg("psf"), py::arg("obs_time"))
            .def("contains", &li::contains, pydocs::DOC_LayeredImage_contains)
            .def("get_science_pixel", &li::get_science_pixel, pydocs::DOC_LayeredImage_get_science_pixel)
            .def("get_variance_pixel", &li::get_variance_pixel, pydocs::DOC_LayeredImage_get_variance_pixel)
            .def("science_pixel_has_data", &li::science_pixel_has_data,
                 pydocs::DOC_LayeredImage_science_pixel_has_data)
            .def("contains",
                 [](li& cls, int i, int j) {
                     return cls.contains({i, j});
                 })
            .def("get_science_pixel",
                 [](li& cls, int i, int j) {
                     return cls.get_science_pixel({i, j});
                 })
            .def("get_variance_pixel",
                 [](li& cls, int i, int j) {
                     return cls.get_variance_pixel({i, j});
                 })
            .def("science_pixel_has_data",
                 [](li& cls, int i, int j) {
                     return cls.science_pixel_has_data({i, j});
                 })
            .def("set_psf", &li::set_psf, pydocs::DOC_LayeredImage_set_psf)
            .def("get_psf", &li::get_psf, py::return_value_policy::reference_internal,
                 pydocs::DOC_LayeredImage_get_psf)
            .def("mask_pixel", &li::mask_pixel, pydocs::DOC_LayeredImage_mask_pixel)
            .def("mask_pixel",
                 [](li& cls, int i, int j) {
                     return cls.mask_pixel({i, j});
                 })
            .def("binarize_mask", &li::binarize_mask, pydocs::DOC_LayeredImage_binarize_mask)
            .def("apply_mask", &li::apply_mask, pydocs::DOC_LayeredImage_apply_mask)
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
