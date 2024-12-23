#include "raw_image.h"

namespace search {
using Index = indexing::Index;
using Point = indexing::Point;

RawImage::RawImage() : width(0), height(0), image() {}

RawImage::RawImage(Image& img) {
    image = std::move(img);
    height = image.rows();
    width = image.cols();
}

RawImage::RawImage(unsigned w, unsigned h, float value) : width(w), height(h) {
    if (value != 0.0f)
        image = Image::Constant(height, width, value);
    else
        image = Image::Zero(height, width);
}

// Copy constructor
RawImage::RawImage(const RawImage& old) noexcept {
    width = old.get_width();
    height = old.get_height();
    image = old.get_image();
}

// Move constructor
RawImage::RawImage(RawImage&& source) noexcept
        : width(source.width), height(source.height), image(std::move(source.image)) {}

// Copy assignment
RawImage& RawImage::operator=(const RawImage& source) noexcept {
    width = source.width;
    height = source.height;
    image = source.image;
    return *this;
}

// Move assignment
RawImage& RawImage::operator=(RawImage&& source) noexcept {
    if (this != &source) {
        width = source.width;
        height = source.height;
        image = std::move(source.image);
    }
    return *this;
}

bool RawImage::l2_allclose(const RawImage& img_b, float atol) const {
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (!pixel_value_valid(image(y, x))) {
                if (pixel_value_valid(img_b.image(y, x))) return false;
            } else if (!pixel_value_valid(img_b.image(y, x))) {
                return false;
            } else {
                if (fabs(image(y, x) - img_b.image(y, x)) > atol) return false;
            }
        }
    }
    return true;
}

void RawImage::replace_masked_values(float value) {
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (!pixel_value_valid(image(y, x))) {
                image(y, x) = value;
            }
        }
    }
}

RawImage RawImage::create_stamp(const Point& p, const int radius, const bool keep_no_data) const {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");

    const int dim = radius * 2 + 1;
    Image stamp = Image::Constant(dim, dim, NO_DATA);

    // Eigen gets unhappy if the stamp does not overlap at all. In this case, skip
    // the computation and leave the entire stamp set to NO_DATA. We use (int) casting here
    // instead of p.to_index() because of how we are defining the pixel grid and the center pixel.
    int idx_j = (int)p.x;
    int idx_i = (int)p.y;
    if ((idx_j + radius >= 0) && (idx_j - radius < (int)width) && (idx_i + radius >= 0) &&
        (idx_i - radius < (int)height)) {
        auto [corner, anchor, w, h] = indexing::anchored_block({idx_i, idx_j}, radius, width, height);
        stamp.block(anchor.i, anchor.j, h, w) = image.block(corner.i, corner.j, h, w);
    }

    RawImage result = RawImage(stamp);
    if (!keep_no_data) result.replace_masked_values(0.0);
    return result;
}

std::array<float, 2> RawImage::compute_bounds() const {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (auto elem : image.reshaped())
        if (pixel_value_valid(elem)) {
            min_val = std::min(min_val, elem);
            max_val = std::max(max_val, elem);
        }

    // Assert that we have seen at least some valid data.
    if (max_val == -FLT_MAX) throw std::runtime_error("No max value found in RawImage.");
    if (min_val == FLT_MAX) throw std::runtime_error("No min value found in RawImage.");

    return {min_val, max_val};
}

void RawImage::convolve_cpu(PSF& psf) {
    Image result = Image::Zero(height, width);

    const int psf_rad = psf.get_radius();
    const float psf_total = psf.get_sum();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Pixels with invalid data (e.g. NO_DATA or NaN) do not change.
            if (!pixel_value_valid(image(y, x))) {
                result(y, x) = image(y, x);
                continue;
            }

            float sum = 0.0;
            float psf_portion = 0.0;
            for (int j = -psf_rad; j <= psf_rad; j++) {
                for (int i = -psf_rad; i <= psf_rad; i++) {
                    if ((x + i >= 0) && (x + i < width) && (y + j >= 0) && (y + j < height)) {
                        float current_pixel = image(y + j, x + i);
                        // note that convention for index access is flipped for PSF
                        if (pixel_value_valid(current_pixel)) {
                            float current_psf = psf.get_value(i + psf_rad, j + psf_rad);
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
    image = std::move(result);
}

#ifdef HAVE_CUDA
// Performs convolution between an image represented as an array of floats
// and a PSF on a GPU device.
extern "C" void deviceConvolve(float* source_img, float* result_img, int width, int height, PSF& psf);
#endif

void RawImage::convolve(PSF& psf) {
#ifdef HAVE_CUDA
    deviceConvolve(image.data(), image.data(), get_width(), get_height(), psf);
#else
    convolve_cpu(psf);
#endif
}

void RawImage::apply_mask(int flags, const RawImage& mask) {
    for (unsigned int j = 0; j < height; ++j) {
        for (unsigned int i = 0; i < width; ++i) {
            int pix_flags = static_cast<int>(mask.image(j, i));
            if ((flags & pix_flags) != 0) {
                image(j, i) = NO_DATA;
            }
        }  // for i
    }      // for j
}

void RawImage::set_all(float value) { image.setConstant(value); }

// The maximum value of the image and return the coordinates.
Index RawImage::find_peak(bool furthest_from_center) const {
    int c_x = width / 2;
    int c_y = height / 2;

    // Initialize the variables for tracking the peak's location.
    Index result = {0, 0};
    float max_val = std::numeric_limits<float>::lowest();
    float dist2 = c_x * c_x + c_y * c_y;

    // Search each pixel for the peak.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float pix_val = image(y, x);
            if (pixel_value_valid(pix_val) && (pix_val > max_val)) {
                max_val = pix_val;
                result.i = y;
                result.j = x;
                dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
            } else if (pixel_value_valid(pix_val) && (pix_val == max_val)) {
                int new_dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
                if ((furthest_from_center && (new_dist2 > dist2)) ||
                    (!furthest_from_center && (new_dist2 < dist2))) {
                    max_val = pix_val;
                    result.i = y;
                    result.j = x;
                    dist2 = new_dist2;
                }
            }
        }  // for x
    }      // for y
    return result;
}

// Find the basic image moments in order to test if stamps have a gaussian shape.
// It computes the moments on the "normalized" image where the minimum
// value has been shifted to zero and the sum of all elements is 1.0.
// Elements with invalid or masked data are treated as zero.
ImageMoments RawImage::find_central_moments() const {
    const uint64_t num_pixels = width * height;
    const int c_x = width / 2;
    const int c_y = height / 2;

    // Set all the moments to zero initially.
    ImageMoments res = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto pixels = image.reshaped();

    // Find the minimum (valid) value to subtract off.
    float min_val = FLT_MAX;
    for (uint64_t p = 0; p < num_pixels; ++p) {
        min_val = (pixel_value_valid(pixels[p]) && (pixels[p] < min_val)) ? pixels[p] : min_val;
    }

    // Find the sum of the zero-shifted (valid) pixels.
    double sum = 0.0;
    for (uint64_t p = 0; p < num_pixels; ++p) {
        sum += pixel_value_valid(pixels[p]) ? (pixels[p] - min_val) : 0.0;
    }
    if (sum == 0.0) return res;

    // Compute the rest of the moments.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int ind = y * width + x;
            float pix_val = pixel_value_valid(pixels[ind]) ? (pixels[ind] - min_val) / sum : 0.0;

            res.m00 += pix_val;
            res.m10 += (x - c_x) * pix_val;
            res.m20 += (x - c_x) * (x - c_x) * pix_val;
            res.m01 += (y - c_y) * pix_val;
            res.m02 += (y - c_y) * (y - c_y) * pix_val;
            res.m11 += (x - c_x) * (y - c_y) * pix_val;
        }
    }

    return res;
}

bool RawImage::center_is_local_max(double flux_thresh, bool local_max) const {
    const uint64_t num_pixels = width * height;
    int c_x = width / 2;
    int c_y = height / 2;
    int c_ind = c_y * width + c_x;

    auto pixels = image.reshaped();
    double center_val = pixels[c_ind];

    // Find the sum of the zero-shifted (valid) pixels.
    double sum = 0.0;
    for (uint64_t p = 0; p < num_pixels; ++p) {
        float pix_val = pixels[p];
        if (p != c_ind && local_max && pix_val >= center_val) {
            return false;
        }
        sum += pixel_value_valid(pixels[p]) ? pix_val : 0.0;
    }
    if (sum == 0.0) return false;
    return center_val / sum >= flux_thresh;
}

// it makes no sense to return RawImage here because there is no
// obstime by definition of operation, but I guess it's out of
// scope for this PR because it requires updating layered_image
// and image stack
RawImage create_median_image(const std::vector<RawImage>& images) {
    unsigned int num_images = images.size();
    if (num_images == 0) throw std::runtime_error("Unable to create median image given 0 images.");

    unsigned int width = images[0].get_width();
    unsigned int height = images[0].get_height();
    for (auto& img : images) {
        assert_sizes_equal(img.get_width(), width, "median images width");
        assert_sizes_equal(img.get_height(), height, "median images height");
    }

    Image result = Image::Zero(height, width);

    std::vector<float> pix_array(num_images);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int num_unmasked = 0;
            for (auto& img : images) {
                // Only used the unmasked array.
                if (img.pixel_has_data({y, x})) {
                    pix_array[num_unmasked] = img.get_pixel({y, x});
                    num_unmasked += 1;
                }
            }

            if (num_unmasked > 0) {
                std::sort(pix_array.begin(), pix_array.begin() + num_unmasked);

                // If we have an even number of elements, take the mean of the two
                // middle ones.
                int median_ind = num_unmasked / 2;
                if ((num_unmasked % 2 == 0) && (median_ind > 0)) {
                    float ave_middle = (pix_array[median_ind] + pix_array[median_ind - 1]) / 2.0;
                    result(y, x) = ave_middle;
                } else {
                    result(y, x) = pix_array[median_ind];
                }
            } else {
                // We use a 0.0 value if there is no data to allow for visualization
                // and value based filtering.
                result(y, x) = 0.0;
            }
        }  // for x
    }      // for y
    return RawImage(result);
}

RawImage create_summed_image(const std::vector<RawImage>& images) {
    unsigned int num_images = images.size();
    if (num_images == 0) throw std::runtime_error("Unable to create summed image given 0 images.");

    unsigned int width = images[0].get_width();
    unsigned int height = images[0].get_height();
    for (auto& img : images) {
        assert_sizes_equal(img.get_width(), width, "summed images width");
        assert_sizes_equal(img.get_height(), height, "summed images height");
    }

    Image result = Image::Zero(height, width);
    for (auto& img : images) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (img.pixel_has_data({y, x})) {
                    result(y, x) += img.get_pixel({y, x});
                }
            }
        }
    }
    return RawImage(result);
}

RawImage create_mean_image(const std::vector<RawImage>& images) {
    unsigned int num_images = images.size();
    if (num_images == 0) throw std::runtime_error("Unable to create mean image given 0 images.");

    unsigned int width = images[0].get_width();
    unsigned int height = images[0].get_height();
    for (auto& img : images) {
        assert_sizes_equal(img.get_width(), width, "mean images width");
        assert_sizes_equal(img.get_height(), height, "mean images height");
    }

    Image result = Image::Zero(height, width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            float count = 0.0;
            for (auto& img : images) {
                if (img.pixel_has_data({y, x})) {
                    count += 1.0;
                    sum += img.get_pixel({y, x});
                }
            }

            if (count > 0.0)
                result(y, x) = sum / count;
            else
                result(y, x) = 0.0;  // use 0 for visualization purposes
        }                            // for x
    }                                // for y
    return RawImage(result);
}

#ifdef Py_PYTHON_H
static void raw_image_bindings(py::module& m) {
    using rie = search::RawImage;

    py::class_<rie>(m, "RawImage", pydocs::DOC_RawImage)
            .def(py::init<>())
            .def(py::init<search::RawImage&>())
            .def(py::init<search::Image&>(), py::arg("img").noconvert(true))
            .def(py::init<unsigned, unsigned, float>(), py::arg("w"), py::arg("h"), py::arg("value") = 0.0f)
            // attributes and properties
            .def_property_readonly("height", &rie::get_height)
            .def_property_readonly("width", &rie::get_width)
            .def_property_readonly("npixels", &rie::get_npixels)
            .def_property("image", py::overload_cast<>(&rie::get_image, py::const_), &rie::set_image)
            .def_property("imref", py::overload_cast<>(&rie::get_image), &rie::set_image)
            // pixel accessors and setters
            .def("get_pixel", &rie::get_pixel, pydocs::DOC_RawImage_get_pixel)
            .def("pixel_has_data", &rie::pixel_has_data, pydocs::DOC_RawImage_pixel_has_data)
            .def("set_pixel", &rie::set_pixel, pydocs::DOC_RawImage_set_pixel)
            .def("mask_pixel", &rie::mask_pixel, pydocs::DOC_RawImage_mask_pixel)
            .def("set_all", &rie::set_all, pydocs::DOC_RawImage_set_all)
            .def("contains_index", py::overload_cast<const indexing::Index&>(&rie::contains, py::const_),
                 pydocs::DOC_RawImage_contains_index)
            .def("contains_point", py::overload_cast<const indexing::Point&>(&rie::contains, py::const_),
                 pydocs::DOC_RawImage_contains_point)
            // python interface adapters (avoids having to construct Index & Point)
            .def("get_pixel",
                 [](rie& cls, int i, int j) {
                     return cls.get_pixel({i, j});
                 })
            .def("pixel_has_data",
                 [](rie& cls, int i, int j) {
                     return cls.pixel_has_data({i, j});
                 })
            .def("set_pixel",
                 [](rie& cls, int i, int j, double val) {
                     cls.set_pixel({i, j}, val);
                 })
            .def("mask_pixel",
                 [](rie& cls, int i, int j) {
                     cls.mask_pixel({i, j});
                 })
            .def("contains_index",
                 [](rie& cls, int i, int j) {
                     return cls.contains(indexing::Index({i, j}));
                 })
            .def("contains_point",
                 [](rie& cls, float x, float y) {
                     return cls.contains(indexing::Point({x, y}));
                 })
            // methods
            .def("l2_allclose", &rie::l2_allclose, pydocs::DOC_RawImage_l2_allclose)
            .def("replace_masked_values", &rie::replace_masked_values, py::arg("value") = 0.0f,
                 pydocs::DOC_RawImage_replace_masked_values)
            .def("compute_bounds", &rie::compute_bounds, pydocs::DOC_RawImage_compute_bounds)
            .def("find_peak", &rie::find_peak, pydocs::DOC_RawImage_find_peak)
            .def("find_central_moments", &rie::find_central_moments,
                 pydocs::DOC_RawImage_find_central_moments)
            .def("center_is_local_max", &rie::center_is_local_max, pydocs::DOC_RawImage_center_is_local_max)
            .def("create_stamp", &rie::create_stamp, pydocs::DOC_RawImage_create_stamp)
            .def("apply_mask", &rie::apply_mask, pydocs::DOC_RawImage_apply_mask)
            .def("convolve_gpu", &rie::convolve, pydocs::DOC_RawImage_convolve_gpu)
            .def("convolve_cpu", &rie::convolve_cpu, pydocs::DOC_RawImage_convolve_cpu)
            // python interface adapters
            .def("create_stamp", [](rie& cls, float x, float y, int radius, bool keep_no_data) {
                return cls.create_stamp({x, y}, radius, keep_no_data);
            });
}
#endif

} /* namespace search */
