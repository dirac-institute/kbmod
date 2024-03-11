#include "layered_image.h"

namespace search {

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

void LayeredImage::set_psf(const PSF& new_psf) { psf = new_psf; }

void LayeredImage::convolve_given_psf(const PSF& given_psf) {
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
    const int num_pixels = get_npixels();
    float* mask_pixels = mask.data();

    for (int i = 0; i < num_pixels; ++i) {
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

void LayeredImage::union_masks(RawImage& new_mask) {
    const int num_pixels = get_npixels();
    if (num_pixels != new_mask.get_npixels()) {
        throw std::runtime_error("Mismatched number of pixels between image and global mask.");
    }

    float* mask_pixels = mask.data();
    float* new_pixels = new_mask.data();
    for (int i = 0; i < num_pixels; ++i) {
        int current_flags = static_cast<int>(mask_pixels[i]);
        int new_flags = static_cast<int>(new_pixels[i]);

        // Use a bitwise OR to keep flags set in the current pixel or the new mask.
        mask_pixels[i] = current_flags | new_flags;
    }
}

void LayeredImage::union_threshold_masking(float thresh) {
    const int num_pixels = get_npixels();
    float* sci_pixels = science.data();
    float* mask_pixels = mask.data();

    for (int i = 0; i < num_pixels; ++i) {
        if (sci_pixels[i] > thresh) {
            // Use a logical OR to preserve all other flags.
            mask_pixels[i] = static_cast<int>(mask_pixels[i]) | 1;
        }
    }
}

/* This implementation of grow_mask is optimized for steps > 1
   (which is how the code is generally used. If you are only
   growing the mask by 1, the extra copy will be a little slower.
*/
void LayeredImage::grow_mask(int steps) {
    ImageI bitmask = ImageI::Constant(height, width, -1);
    bitmask = (mask.get_image().array() > 0).select(0, bitmask);

    for (int itr = 1; itr <= steps; ++itr) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                if (bitmask(j, i) == -1) {
                    if (((j - 1 >= 0) && (bitmask(j - 1, i) == itr - 1)) ||
                        ((i - 1 >= 0) && (bitmask(j, i - 1) == itr - 1)) ||
                        ((j + 1 < height) && (bitmask(j + 1, i) == itr - 1)) ||
                        ((i + 1 < width) && (bitmask(j, i + 1) == itr - 1))) {
                        bitmask(j, i) = itr;
                    }
                }
            }  // for i
        }      // for j
    }          // for step

    // Overwrite the mask with the expanded one.
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            mask.set_pixel({j, i}, (bitmask(j, i) == -1) ? 0 : 1);
        }
    }
}

void LayeredImage::subtract_template(RawImage& sub_template) {
    assert(get_height() == sub_template.get_height() && get_width() == sub_template.get_width());
    const int num_pixels = get_npixels();

    float* sci_pixels = science.data();
    float* tem_pixels = sub_template.data();
    for (unsigned i = 0; i < num_pixels; ++i) {
        if (pixel_value_valid(sci_pixels[i]) && pixel_value_valid(tem_pixels[i])) {
            sci_pixels[i] -= tem_pixels[i];
        }
    }
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
        if (pixel_value_valid(var_pix) && var_pix != 0.0 && pixel_value_valid(sci_array[p])) {
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
        if (pixel_value_valid(var_pix) && var_pix != 0.0) {
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
            .def(py::init<const ri&, const ri&, const ri&, pf&>())
            .def("contains", &li::contains, pydocs::DOC_LayeredImage_cointains)
            .def("get_science_pixel", &li::get_science_pixel, pydocs::DOC_LayeredImage_get_science_pixel)
            .def("get_variance_pixel", &li::get_variance_pixel, pydocs::DOC_LayeredImage_get_variance_pixel)
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
            .def("union_masks", &li::union_masks, pydocs::DOC_LayeredImage_union_masks)
            .def("union_threshold_masking", &li::union_threshold_masking,
                 pydocs::DOC_LayeredImage_union_threshold_masking)
            .def("sub_template", &li::subtract_template, pydocs::DOC_LayeredImage_sub_template)
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
