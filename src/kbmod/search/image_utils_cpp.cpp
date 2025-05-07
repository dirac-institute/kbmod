/* Helper functions for testing functions in the .cu files from Python. */

#include "image_utils_cpp.h"
#include "logging.h"
#include "pydocs/image_utils_cpp_docs.h"

namespace search {

// -------------------------------------------------------
// --- Convolution Functions -----------------------------
// -------------------------------------------------------

#ifdef HAVE_CUDA
// Performs convolution between an image represented as an array of floats
// and a PSF on a GPU device.
extern "C" void deviceConvolve(float* source_img, float* result_img, int width, int height, float* psf_kernel,
                               int psf_radius);
#endif

Image convolve_image_cpu(Image& img, Image& psf) {
    const uint64_t img_height = img.rows();
    const uint64_t img_width = img.cols();
    Image result = Image::Zero(img_height, img_width);

    const int psf_num_rows = psf.rows();
    const int psf_num_cols = psf.cols();
    const int psf_rad = (int)((psf_num_rows - 1) / 2);

    // Compute the sum of the PSF.
    float psf_total = 0.0;
    for (int r = 0; r < psf_num_rows; ++r) {
        for (int c = 0; c < psf_num_cols; ++c) {
            psf_total += psf(r, c);
        }
    }

    // Do the convolution using nested loops.
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            // Pixels with invalid data (e.g. NO_DATA or NaN) do not change.
            if (!pixel_value_valid(img(y, x))) {
                result(y, x) = img(y, x);
                continue;
            }

            float sum = 0.0;
            float psf_portion = 0.0;
            for (int j = -psf_rad; j <= psf_rad; j++) {
                for (int i = -psf_rad; i <= psf_rad; i++) {
                    if ((x + i >= 0) && (x + i < img_width) && (y + j >= 0) && (y + j < img_height)) {
                        float current_pixel = img(y + j, x + i);
                        if (pixel_value_valid(current_pixel)) {
                            float current_psf = psf(j + psf_rad, i + psf_rad);
                            psf_portion += current_psf;
                            sum += current_pixel * current_psf;
                        }
                    }
                }  // for i
            }      // for j
            if (psf_portion == 0) {
                result(y, x) = NO_DATA;
            } else {
                result(y, x) = (sum * psf_total) / psf_portion;
            }
        }  // for x
    }      // for y
    return result;
}

Image convolve_image_gpu(Image& img, Image& psf) {
    const uint64_t img_height = img.rows();
    const uint64_t img_width = img.cols();
    Image result = Image::Zero(img_height, img_width);

#ifdef HAVE_CUDA
    // Extract the PSF kernel into a flat array. There is probably a better
    // way to do this via the Eigen library.
    int num_rows = psf.rows();
    int num_cols = psf.cols();
    std::vector<float> psf_vals(num_rows * num_cols);
    int idx = 0;
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            psf_vals[idx] = psf(r, c);
            ++idx;
        }
    }

    int radius = (num_rows - 1) / 2;
    deviceConvolve(img.data(), result.data(), img_width, img_height, psf_vals.data(), radius);
#else
    throw std::runtime_error("Unable to perform convolve_image_gpu() without GPU.");
#endif

    return result;
}

Image convolve_image(Image& image, Image& psf) {
#ifdef HAVE_CUDA
    Image result = convolve_image_gpu(image, psf);
#else
    Image result = convolve_image_cpu(image, psf);
#endif

    return result;
}

Image square_psf(Image& given_psf) {
    // Make a copy of the PSF.
    Image psf_sq = given_psf;

    for (int r = 0; r < given_psf.rows(); ++r) {
        for (int c = 0; c < given_psf.cols(); ++c) {
            psf_sq(r, c) = given_psf(r, c) * given_psf(r, c);
        }
    }
    return psf_sq;
}

// -------------------------------------------------------
// --- Functions for Psi and Phi Generation --------------
// -------------------------------------------------------

Image generate_psi_image(Image& sci, Image& var, Image& psf) {
    const uint64_t height = sci.rows();
    const uint64_t width = sci.cols();
    const uint64_t num_pixels = height * width;
    if ((height != var.rows()) || (width != var.cols())) {
        throw std::runtime_error("Science and Variance images must be the same dimensions.");
    }

    Image result = Image::Zero(height, width);
    float* result_arr = result.data();
    float* sci_array = sci.data();
    float* var_array = var.data();

    // Set each of the result pixels.
    for (uint64_t p = 0; p < num_pixels; ++p) {
        float var_pix = var_array[p];
        if (std::isfinite(var_pix) && var_pix != 0.0 && std::isfinite(sci_array[p])) {
            result_arr[p] = sci_array[p] / var_pix;
        } else {
            result_arr[p] = NO_DATA;
        }
    }

    // Convolve with the PSF.
    return convolve_image(result, psf);
}

Image generate_phi_image(Image& var, Image& psf) {
    const uint64_t height = var.rows();
    const uint64_t width = var.cols();
    const uint64_t num_pixels = height * width;

    Image result = Image::Zero(height, width);
    float* result_arr = result.data();
    float* var_array = var.data();

    // Set each of the result pixels.
    for (uint64_t p = 0; p < num_pixels; ++p) {
        float var_pix = var_array[p];
        if (std::isfinite(var_pix) && var_pix != 0.0) {
            result_arr[p] = 1.0 / var_pix;
        } else {
            result_arr[p] = NO_DATA;
        }
    }

    // Convolve with the PSF squared.
    Image psfsq = square_psf(psf);  // Copy
    return convolve_image(result, psfsq);
}

#ifdef Py_PYTHON_H
static void image_utils_cpp(py::module& m) {
    m.def("convolve_image_cpu", &search::convolve_image_cpu, py::arg("image").noconvert(true),
          py::arg("psf").noconvert(true), pydocs::DOC_image_utils_cpp_convolve_cpu);
    m.def("convolve_image_gpu", &search::convolve_image_gpu, py::arg("image").noconvert(true),
          py::arg("psf").noconvert(true), pydocs::DOC_image_utils_cpp_convolve_gpu);
    m.def("convolve_image", &search::convolve_image, py::arg("image").noconvert(true),
          py::arg("psf").noconvert(true), pydocs::DOC_image_utils_cpp_convolve);
    m.def("square_psf", &search::square_psf, py::arg("given_psf").noconvert(true),
          pydocs::DOC_image_utils_square_psf);
    m.def("generate_psi_image", &search::generate_psi_image, py::arg("sci").noconvert(true),
          py::arg("var").noconvert(true), py::arg("psf").noconvert(true),
          pydocs::DOC_image_utils_generate_psi_image);
    m.def("generate_phi_image", &search::generate_phi_image, py::arg("var").noconvert(true),
          py::arg("psf").noconvert(true), pydocs::DOC_image_utils_generate_phi_image);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
