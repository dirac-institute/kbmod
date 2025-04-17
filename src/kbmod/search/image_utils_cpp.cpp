/* Helper functions for testing functions in the .cu files from Python. */

#include "image_utils_cpp.h"
#include "logging.h"
#include "pydocs/image_utils_cpp_docs.h"

namespace search {

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

#ifdef Py_PYTHON_H
static void image_utils_cpp(py::module& m) {
    m.def("convolve_image_cpu", &search::convolve_image_cpu, pydocs::DOC_image_utils_cpp_convolve_cpu);
    m.def("convolve_image_gpu", &search::convolve_image_gpu, pydocs::DOC_image_utils_cpp_convolve_gpu);
    m.def("convolve_image", &search::convolve_image, pydocs::DOC_image_utils_cpp_convolve);
}
#endif /* Py_PYTHON_H */

} /* namespace search */
