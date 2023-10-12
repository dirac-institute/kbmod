#include "raw_image.h"

namespace py = pybind11;

namespace search {
#ifdef HAVE_CUDA
// Performs convolution between an image represented as an array of floats
// and a PSF on a GPU device.
extern "C" void deviceConvolve(float* source_img, float* result_img, int width, int height, float* psf_kernel,
                               int psf_size, int psf_dim, int psf_radius, float psf_sum);
#endif

RawImage::RawImage() : width(0), height(0), obstime(-1.0) { pixels = std::vector<float>(); }

// Copy constructor
RawImage::RawImage(const RawImage& old) {
    width = old.get_width();
    height = old.get_height();
    pixels = old.get_pixels();
    obstime = old.get_obstime();
}

// Copy assignment
RawImage& RawImage::operator=(const RawImage& source) {
    width = source.width;
    height = source.height;
    pixels = source.pixels;
    obstime = source.obstime;
    return *this;
}

// Move constructor
RawImage::RawImage(RawImage&& source)
        : width(source.width),
          height(source.height),
          obstime(source.obstime),
          pixels(std::move(source.pixels)) {}

// Move assignment
RawImage& RawImage::operator=(RawImage&& source) {
    if (this != &source) {
        width = source.width;
        height = source.height;
        pixels = std::move(source.pixels);
        obstime = source.obstime;
    }
    return *this;
}

RawImage::RawImage(unsigned w, unsigned h) : height(h), width(w), obstime(-1.0), pixels(w * h) {}

RawImage::RawImage(unsigned w, unsigned h, const std::vector<float>& pix)
        : width(w), height(h), obstime(-1.0), pixels(pix) {
    assert(w * h == pix.size());
}

bool RawImage::approx_equal(const RawImage& img_b, float atol) const {
    if ((width != img_b.width) || (height != img_b.height)) return false;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float p1 = get_pixel(x, y);
            float p2 = img_b.get_pixel(x, y);

            // NO_DATA values must match exactly.
            if ((p1 == NO_DATA) && (p2 != NO_DATA)) return false;
            if ((p1 != NO_DATA) && (p2 == NO_DATA)) return false;

            // Other values match within an absolute tolerance.
            if (fabs(p1 - p2) > atol) return false;
        }
    }
    return true;
}

void RawImage::load_time_from_file(fitsfile* fptr) {
    int mjd_status = 0;
    obstime = -1.0;

    // Read image observation time, trying the MJD field first and DATE-AVG second.
    // Ignore error if does not exist.
    if (fits_read_key(fptr, TDOUBLE, "MJD", &obstime, NULL, &mjd_status)) {
        if (fits_read_key(fptr, TDOUBLE, "DATE-AVG", &obstime, NULL, &mjd_status)) {
            obstime = -1.0;
        }
    }
}

// Load the image data from a specific layer of a FITS file.
void RawImage::load_from_file(const std::string& file_path, int layer_num) {
    // Open the file's header and read in the obstime and the dimensions.
    fitsfile* fptr;
    int status = 0;
    int file_not_found;
    int nullval = 0;
    int anynull = 0;

    // Open the correct layer to extract the RawImage.
    std::string layerPath = file_path + "[" + std::to_string(layer_num) + "]";
    if (fits_open_file(&fptr, layerPath.c_str(), READONLY, &status)) {
        fits_report_error(stderr, status);
        throw std::runtime_error("Could not open FITS file to read RawImage");
    }

    // Read image dimensions.
    long dimensions[2];
    if (fits_read_keys_lng(fptr, "NAXIS", 1, 2, dimensions, &file_not_found, &status))
        fits_report_error(stderr, status);
    width = dimensions[0];
    height = dimensions[1];

    // Read in the image.
    pixels = std::vector<float>(width * height);
    if (fits_read_img(fptr, TFLOAT, 1, get_npixels(), &nullval, pixels.data(), &anynull, &status))
        fits_report_error(stderr, status);

    load_time_from_file(fptr);
    if (fits_close_file(fptr, &status)) fits_report_error(stderr, status);

    // If we are reading from a sublayer and did not find a time, try the overall header.
    if (obstime < 0.0) {
        if (fits_open_file(&fptr, file_path.c_str(), READONLY, &status))
            throw std::runtime_error("Could not open FITS file to read RawImage");
        load_time_from_file(fptr);
    }
}

void RawImage::save_to_file(const std::string& filename) {
    fitsfile* fptr;
    int status = 0;
    long naxes[2] = {0, 0};

    fits_create_file(&fptr, filename.c_str(), &status);

    // If we are unable to create the file, check if it already exists
    // and, if so, delete it and retry the create.
    if (status == 105) {
        status = 0;
        fits_open_file(&fptr, filename.c_str(), READWRITE, &status);
        if (status == 0) {
            fits_delete_file(fptr, &status);
            fits_create_file(&fptr, filename.c_str(), &status);
        }
    }

    // Create the primary array image (32-bit float pixels)
    long dimensions[2];
    dimensions[0] = width;
    dimensions[1] = height;
    fits_create_img(fptr, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
    fits_report_error(stderr, status);

    /* Write the array of floats to the image */
    fits_write_img(fptr, TFLOAT, 1, get_npixels(), pixels.data(), &status);
    fits_report_error(stderr, status);

    // Add the basic header data.
    fits_update_key(fptr, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_report_error(stderr, status);

    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
}

void RawImage::append_layer_to_file(const std::string& filename) {
    int status = 0;
    fitsfile* f;

    // Check that we can open the file.
    if (fits_open_file(&f, filename.c_str(), READWRITE, &status)) {
        fits_report_error(stderr, status);
        throw std::runtime_error("Unable to open FITS file for appending.");
    }

    // This appends a layer (extension) if the file exists)
    /* Create the primary array image (32-bit float pixels) */
    long dimensions[2];
    dimensions[0] = width;
    dimensions[1] = height;
    fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
    fits_report_error(stderr, status);

    /* Write the array of floats to the image */
    fits_write_img(f, TFLOAT, 1, get_npixels(), pixels.data(), &status);
    fits_report_error(stderr, status);

    // Save the image time in the header.
    fits_update_key(f, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_report_error(stderr, status);

    fits_close_file(f, &status);
    fits_report_error(stderr, status);
}

RawImage RawImage::create_stamp(float x, float y, int radius, bool interpolate, bool keep_no_data) const {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");

    int dim = radius * 2 + 1;
    RawImage stamp(dim, dim);
    for (int yoff = 0; yoff < dim; ++yoff) {
        for (int xoff = 0; xoff < dim; ++xoff) {
            float pix_val;
            if (interpolate)
                pix_val = get_pixel_interp(x + static_cast<float>(xoff - radius),
                                           y + static_cast<float>(yoff - radius));
            else
                pix_val = get_pixel(static_cast<int>(x) + xoff - radius, static_cast<int>(y) + yoff - radius);
            if ((pix_val == NO_DATA) && !keep_no_data) pix_val = 0.0;
            stamp.set_pixel(xoff, yoff, pix_val);
        }
    }

    stamp.set_obstime(obstime);
    return stamp;
}

void RawImage::convolve_cpu(const PSF& psf) {
    std::vector<float> result(width * height, 0.0);
    const int psf_rad = psf.get_radius();
    const float psf_total = psf.get_sum();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Pixels with NO_DATA remain NO_DATA.
            if (pixels[y * width + x] == NO_DATA) {
                result[y * width + x] = NO_DATA;
                continue;
            }

            float sum = 0.0;
            float psf_portion = 0.0;
            for (int j = -psf_rad; j <= psf_rad; j++) {
                for (int i = -psf_rad; i <= psf_rad; i++) {
                    if ((x + i >= 0) && (x + i < width) && (y + j >= 0) && (y + j < height)) {
                        float current_pixel = pixels[(y + j) * width + (x + i)];
                        if (current_pixel != NO_DATA) {
                            float current_psf = psf.get_value(i + psf_rad, j + psf_rad);
                            psf_portion += current_psf;
                            sum += current_pixel * current_psf;
                        }
                    }
                }
            }
            result[y * width + x] = (sum * psf_total) / psf_portion;
        }
    }

    // Copy the data into the pixels vector.
    const int npixels = get_npixels();
    for (int i = 0; i < npixels; ++i) {
        pixels[i] = result[i];
    }
}

void RawImage::convolve(PSF psf) {
#ifdef HAVE_CUDA
    deviceConvolve(pixels.data(), pixels.data(), get_width(), get_height(), psf.data(), psf.get_size(),
                   psf.get_dim(), psf.get_radius(), psf.get_sum());
#else
    convolve_cpu(psf);
#endif
}

void RawImage::apply_mask(int flags, const std::vector<int>& exceptions, const RawImage& mask) {
    const std::vector<float>& mask_pix = mask.get_pixels();
    const int num_pixels = get_npixels();
    assert(num_pixels == mask.get_npixels());
    for (unsigned int p = 0; p < num_pixels; ++p) {
        int pix_flags = static_cast<int>(mask_pix[p]);
        bool is_exception = false;
        for (auto& e : exceptions) is_exception = is_exception || e == pix_flags;
        if (!is_exception && ((flags & pix_flags) != 0)) pixels[p] = NO_DATA;
    }
}

/* This implementation of grow_mask is optimized for steps > 1
   (which is how the code is generally used. If you are only
   growing the mask by 1, the extra copy will be a little slower.
*/
void RawImage::grow_mask(int steps) {
    const int num_pixels = width * height;

    // Set up the initial masked vector that stores the number of steps
    // each pixel is from a masked pixel in the original image.
    std::vector<int> masked(num_pixels, -1);
    for (int i = 0; i < num_pixels; ++i) {
        if (pixels[i] == NO_DATA) masked[i] = 0;
    }

    // Grow out the mask one for each step.
    for (int itr = 1; itr <= steps; ++itr) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int center = width * y + x;
                if (masked[center] == -1) {
                    // Mask pixels that are adjacent to a pixel masked during
                    // the last iteration only.
                    if ((x + 1 < width && masked[center + 1] == itr - 1) ||
                        (x - 1 >= 0 && masked[center - 1] == itr - 1) ||
                        (y + 1 < height && masked[center + width] == itr - 1) ||
                        (y - 1 >= 0 && masked[center - width] == itr - 1)) {
                        masked[center] = itr;
                    }
                }
            }
        }
    }

    // Mask the pixels in the image.
    for (std::size_t i = 0; i < num_pixels; ++i) {
        if (masked[i] > -1) {
            pixels[i] = NO_DATA;
        }
    }
}

std::vector<float> RawImage::bilinear_interp(float x, float y) const {
    // Linear interpolation
    // Find the 4 pixels (aPix, bPix, cPix, dPix)
    // that the corners (a, b, c, d) of the
    // new pixel land in, and blend into those

    // Returns a vector with 4 pixel locations
    // and their interpolation value

    // Top right
    float ax = x + 0.5;
    float ay = y + 0.5;
    float a_px = floor(ax);
    float a_py = floor(ay);
    float a_amt = (ax - a_px) * (ay - a_py);

    // Bottom right
    float bx = x + 0.5;
    float by = y - 0.5;
    float b_px = floor(bx);
    float b_py = floor(by);
    float b_amt = (bx - b_px) * (b_py + 1.0 - by);

    // Bottom left
    float cx = x - 0.5;
    float cy = y - 0.5;
    float c_px = floor(cx);
    float c_py = floor(cy);
    float c_amt = (c_px + 1.0 - cx) * (c_py + 1.0 - cy);

    // Top left
    float dx = x - 0.5;
    float dy = y + 0.5;
    float d_px = floor(dx);
    float d_py = floor(dy);
    float d_amt = (d_px + 1.0 - dx) * (dy - d_py);

    // make sure the right amount has been distributed
    float diff = std::abs(a_amt + b_amt + c_amt + d_amt - 1.0);
    if (diff > 0.01) std::cout << "warning: bilinear_interpSum == " << diff << "\n";
    return {a_px, a_py, a_amt, b_px, b_py, b_amt, c_px, c_py, c_amt, d_px, d_py, d_amt};
}

void RawImage::add_pixel_interp(float x, float y, float value) {
    // Interpolation values
    std::vector<float> iv = bilinear_interp(x, y);

    add_to_pixel(iv[0], iv[1], value * iv[2]);
    add_to_pixel(iv[3], iv[4], value * iv[5]);
    add_to_pixel(iv[6], iv[7], value * iv[8]);
    add_to_pixel(iv[9], iv[10], value * iv[11]);
}

void RawImage::add_to_pixel(float fx, float fy, float value) {
    assert(fx - floor(fx) == 0.0 && fy - floor(fy) == 0.0);
    int x = static_cast<int>(fx);
    int y = static_cast<int>(fy);
    if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] += value;
}

float RawImage::get_pixel_interp(float x, float y) const {
    if ((x < 0.0 || y < 0.0) || (x > static_cast<float>(width) || y > static_cast<float>(height)))
        return NO_DATA;
    std::vector<float> iv = bilinear_interp(x, y);
    float a = get_pixel(iv[0], iv[1]);
    float b = get_pixel(iv[3], iv[4]);
    float c = get_pixel(iv[6], iv[7]);
    float d = get_pixel(iv[9], iv[10]);
    float interpSum = 0.0;
    float total = 0.0;
    if (a != NO_DATA) {
        interpSum += iv[2];
        total += a * iv[2];
    }
    if (b != NO_DATA) {
        interpSum += iv[5];
        total += b * iv[5];
    }
    if (c != NO_DATA) {
        interpSum += iv[8];
        total += c * iv[8];
    }
    if (d != NO_DATA) {
        interpSum += iv[11];
        total += d * iv[11];
    }
    if (interpSum == 0.0) {
        return NO_DATA;
    } else {
        return total / interpSum;
    }
}

void RawImage::set_all_pix(float value) {
    for (auto& p : pixels) p = value;
}

std::array<float, 2> RawImage::compute_bounds() const {
    const int num_pixels = get_npixels();
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (unsigned p = 0; p < num_pixels; ++p) {
        if (pixels[p] != NO_DATA) {
            min_val = std::min(min_val, pixels[p]);
            max_val = std::max(max_val, pixels[p]);
        }
    }

    // Assert that we have seen at least some valid data.
    assert(max_val != -FLT_MAX);
    assert(min_val != FLT_MAX);

    // Set and return the result array.
    std::array<float, 2> res;
    res[0] = min_val;
    res[1] = max_val;
    return res;
}

// The maximum value of the image and return the coordinates.
PixelPos RawImage::find_peak(bool furthest_from_center) const {
    int c_x = width / 2;
    int c_y = height / 2;

    // Initialize the variables for tracking the peak's location.
    PixelPos result = {0, 0};
    float max_val = pixels[0];
    float dist2 = c_x * c_x + c_y * c_y;

    // Search each pixel for the peak.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float pix_val = pixels[y * width + x];
            if (pix_val > max_val) {
                max_val = pix_val;
                result.x = x;
                result.y = y;
                dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
            } else if (pix_val == max_val) {
                int new_dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
                if ((furthest_from_center && (new_dist2 > dist2)) ||
                    (!furthest_from_center && (new_dist2 < dist2))) {
                    max_val = pix_val;
                    result.x = x;
                    result.y = y;
                    dist2 = new_dist2;
                }
            }
        }
    }

    return result;
}

// Find the basic image moments in order to test if stamps have a gaussian shape.
// It computes the moments on the "normalized" image where the minimum
// value has been shifted to zero and the sum of all elements is 1.0.
// Elements with NO_DATA are treated as zero.
ImageMoments RawImage::find_central_moments() const {
    const int num_pixels = width * height;
    const int c_x = width / 2;
    const int c_y = height / 2;

    // Set all the moments to zero initially.
    ImageMoments res = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Find the min (non-NO_DATA) value to subtract off.
    float min_val = FLT_MAX;
    for (int p = 0; p < num_pixels; ++p) {
        min_val = ((pixels[p] != NO_DATA) && (pixels[p] < min_val)) ? pixels[p] : min_val;
    }

    // Find the sum of the zero-shifted (non-NO_DATA) pixels.
    double sum = 0.0;
    for (int p = 0; p < num_pixels; ++p) {
        sum += (pixels[p] != NO_DATA) ? (pixels[p] - min_val) : 0.0;
    }
    if (sum == 0.0) return res;

    // Compute the rest of the moments.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int ind = y * width + x;
            float pix_val = (pixels[ind] != NO_DATA) ? (pixels[ind] - min_val) / sum : 0.0;

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

RawImage create_median_image(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) assert(img.get_width() == width and img.get_height() == height);

    RawImage result = RawImage(width, height);
    std::vector<float> pix_array(num_images);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int num_unmasked = 0;
            for (int i = 0; i < num_images; ++i) {
                // Only used the unmasked pixels.
                float pix_val = images[i].get_pixel(x, y);
                if ((pix_val != NO_DATA) && (!std::isnan(pix_val))) {
                    pix_array[num_unmasked] = pix_val;
                    num_unmasked += 1;
                }
            }

            if (num_unmasked > 0) {
                std::sort(pix_array.begin(), pix_array.begin() + num_unmasked);

                // If we have an even number of elements, take the mean of the two
                // middle ones.
                int median_ind = num_unmasked / 2;
                if (num_unmasked % 2 == 0) {
                    float ave_middle = (pix_array[median_ind] + pix_array[median_ind - 1]) / 2.0;
                    result.set_pixel(x, y, ave_middle);
                } else {
                    result.set_pixel(x, y, pix_array[median_ind]);
                }
            } else {
                // We use a 0.0 value if there is no data to allow for visualization
                // and value based filtering.
                result.set_pixel(x, y, 0.0);
            }
        }
    }

    return result;
}

RawImage create_summed_image(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) assert(img.get_width() == width and img.get_height() == height);

    RawImage result = RawImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            for (int i = 0; i < num_images; ++i) {
                float pix_val = images[i].get_pixel(x, y);
                if ((pix_val == NO_DATA) || (std::isnan(pix_val))) pix_val = 0.0;
                sum += pix_val;
            }
            result.set_pixel(x, y, sum);
        }
    }

    return result;
}

RawImage create_mean_image(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) assert(img.get_width() == width and img.get_height() == height);

    RawImage result = RawImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            float count = 0.0;
            for (int i = 0; i < num_images; ++i) {
                float pix_val = images[i].get_pixel(x, y);
                if ((pix_val != NO_DATA) && (!std::isnan(pix_val))) {
                    count += 1.0;
                    sum += pix_val;
                }
            }

            if (count > 0.0) {
                result.set_pixel(x, y, sum / count);
            } else {
                // We use a 0.0 value if there is no data to allow for visualization
                // and value based filtering.
                result.set_pixel(x, y, 0.0);
            }
        }
    }

    return result;
}

#ifdef Py_PYTHON_H
RawImage::RawImage(pybind11::array_t<float> arr) {
    obstime = -1.0;
    set_array(arr);
}

void RawImage::set_array(pybind11::array_t<float>& arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2) throw std::runtime_error("Array must have 2 dimensions.");

    width = info.shape[1];
    height = info.shape[0];
    float* pix = static_cast<float*>(info.ptr);

    pixels = std::vector<float>(pix, pix + get_npixels());
}

static void raw_image_bindings(py::module& m) {
    using ri = search::RawImage;

    py::class_<ri>(m, "RawImage", py::buffer_protocol(), pydocs::DOC_RawImage)
            .def_buffer([](ri& m) -> py::buffer_info {
                return py::buffer_info(m.data(), sizeof(float), py::format_descriptor<float>::format(), 2,
                                       {m.get_height(), m.get_width()},
                                       {sizeof(float) * m.get_width(), sizeof(float)});
            })
            .def(py::init<int, int>())
            .def(py::init<const ri&>())
            .def(py::init<py::array_t<float>>())
            .def("get_height", &ri::get_height, pydocs::DOC_RawImage_get_height)
            .def("get_width", &ri::get_width, pydocs::DOC_RawImage_get_width)
            .def("get_npixels", &ri::get_npixels, pydocs::DOC_RawImage_get_npixels)
            .def("get_all_pixels", &ri::get_pixels, pydocs::DOC_RawImage_get_all_pixels)
            .def("set_array", &ri::set_array, pydocs::DOC_RawImage_set_array)
            .def("get_obstime", &ri::get_obstime, pydocs::DOC_RawImage_get_obstime)
            .def("set_obstime", &ri::set_obstime, pydocs::DOC_RawImage_set_obstime)
            .def("approx_equal", &ri::approx_equal, pydocs::DOC_RawImage_approx_equal)
            .def("compute_bounds", &ri::compute_bounds, pydocs::DOC_RawImage_compute_bounds)
            .def("find_peak", &ri::find_peak, pydocs::DOC_RawImage_find_peak)
            .def("find_central_moments", &ri::find_central_moments, pydocs::DOC_RawImage_find_central_moments)
            .def("create_stamp", &ri::create_stamp, pydocs::DOC_RawImage_create_stamp)
            .def("set_pixel", &ri::set_pixel, pydocs::DOC_RawImage_set_pixel)
            .def("add_pixel", &ri::add_to_pixel, pydocs::DOC_RawImage_add_pixel)
            .def("add_pixel_interp", &ri::add_pixel_interp, pydocs::DOC_RawImage_add_pixel_interp)
            .def("apply_mask", &ri::apply_mask, pydocs::DOC_RawImage_apply_mask)
            .def("grow_mask", &ri::grow_mask, pydocs::DOC_RawImage_grow_mask)
            .def("pixel_has_data", &ri::pixel_has_data, pydocs::DOC_RawImage_pixel_has_data)
            .def("set_all", &ri::set_all_pix, pydocs::DOC_RawImage_set_all)
            .def("get_pixel", &ri::get_pixel, pydocs::DOC_RawImage_get_pixel)
            .def("get_pixel_interp", &ri::get_pixel_interp, pydocs::DOC_RawImage_get_pixel_interp)
            .def("convolve", &ri::convolve, pydocs::DOC_RawImage_convolve)
            .def("convolve_cpu", &ri::convolve_cpu, pydocs::DOC_RawImage_convolve_cpu)
            .def("load_fits", &ri::load_from_file, pydocs::DOC_RawImage_load_fits)
            .def("save_fits", &ri::save_to_file, pydocs::DOC_RawImage_save_fits)
            .def("append_fits_layer", &ri::append_layer_to_file, pydocs::DOC_RawImage_append_fits_layer);
}
#endif

} /* namespace search */
