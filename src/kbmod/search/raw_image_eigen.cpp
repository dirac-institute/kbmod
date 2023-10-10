#include "raw_image_eigen.h"


namespace search {
  RawImageEigen::RawImageEigen()
    : width(0),
      height(0),
      obstime(-1.0),
      image() {}


  RawImageEigen::RawImageEigen(Image& img, double obs_time)  {
    image = std::move(img);
    height = image.rows();
    width = image.cols();
    obstime = obs_time;
  }


  RawImageEigen::RawImageEigen(unsigned h, unsigned w, float value, double obs_time)
    : height(h),
      width(w),
      obstime(obs_time) {
    if (value != 0.0f)
      image = Image::Constant(height, width, value);
    else
      image = Image::Zero(height, width);
  }


  // Copy constructor
  RawImageEigen::RawImageEigen(const RawImageEigen& old) {
    width = old.get_width();
    height = old.get_height();
    image = old.image;
    obstime = old.get_obstime();
  }


  // Move constructor
  RawImageEigen::RawImageEigen(RawImageEigen&& source)
    : width(source.width),
      height(source.height),
      obstime(source.obstime),
      image(std::move(source.image)) {}


  // Copy assignment
  RawImageEigen& RawImageEigen::operator=(const RawImageEigen& source) {
    width = source.width;
    height = source.height;
    image = source.image;
    obstime = source.obstime;
    return *this;
  }


  // Move assignment
  RawImageEigen& RawImageEigen::operator=(RawImageEigen&& source) {
    if (this != &source) {
      width = source.width;
      height = source.height;
      image = std::move(source.image);
      obstime = source.obstime;
    }
    return *this;
  }


  bool RawImageEigen::l2_allclose(const RawImageEigen& img_b, float atol) const {
    // https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
    return image.isApprox(img_b.image, atol);
  }


  std::array<float, 12> RawImageEigen::get_interp_neighbors_and_weights(float y, float x) const {
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


  float RawImageEigen::interpolate(const float y, const float x) const {
    if ((x < 0.0 || y < 0.0) || (x > static_cast<float>(width) || y > static_cast<float>(height)))
      return NO_DATA;

    auto [a_x, a_y, w_a, b_x, b_y, w_b, c_x, c_y, w_c, d_x, d_y, w_d] = get_interp_neighbors_and_weights(y, x);

    float a = get_pixel(a_y, a_x);
    float b = get_pixel(b_y, b_x);
    float c = get_pixel(c_y, c_x);
    float d = get_pixel(d_y, d_x);
    float interpSum = 0.0;
    float total = 0.0;
    if (a != NO_DATA) {
      interpSum += w_a;
      total += a * w_a;
    }
    if (b != NO_DATA) {
      interpSum += w_b;
      total += b * w_b;
    }
    if (c != NO_DATA) {
      interpSum += w_c;
      total += c * w_c;
    }
    if (d != NO_DATA) {
      interpSum += w_d;
      total += d * w_d;
    }
    if (interpSum == 0.0) {
      return NO_DATA;
    } else {
      return total / interpSum;
    }
  }


  Image RawImageEigen::create_stamp(const float y, const float x, const int radius,
                                    const bool interpolate, const bool keep_no_data) const {
    if (radius < 0)
      throw std::runtime_error("stamp radius must be at least 0");

    const int dim = radius*2 + 1;
    Index idx(x, y, false);
    auto [i, dimx, j, dimy] = idx.centered_block(radius, width, height);
    Image stamp = image.block(j, i, dimy, dimx);

    if (interpolate) {
      for (int yoff = 0; yoff < dim; ++yoff) {
        for (int xoff = 0; xoff < dim; ++xoff) {
          stamp(yoff, xoff) = this->interpolate(y + static_cast<float>(yoff - radius), x + static_cast<float>(xoff - radius));
        }
      }
    }

    return stamp;
  }


  void RawImageEigen::add(const float x, const float y, const float value) {
    // In the original implementation this just does array[x, y] += value
    // even though that doesn't really make any sense? The cast is probably
    // implicit. This is likely slower than before, because the floor is used
    // not a cast. Floor is more correct, but I added the constructor nonetheless
    Index p(x, y, false);
    if (contains(p))
      image(p.i, p.j) += value;
  }


  void RawImageEigen::add(const int x, const int y, const float value) {
    Index p(x, y);
    if (contains(p))
      image(p.i, p.j) += value;
  }


  void RawImageEigen::add(const Index p, const float value) {
    if (contains(p))
      image(p.i, p.j) += value;
  }


  void RawImageEigen::interpolated_add(const float x, const float y, const float value) {
    // interpolation weights and neighbors
    auto [a_x, a_y, w_a, b_x, b_y, w_b, c_x, c_y, w_c, d_x, d_y, w_d] = get_interp_neighbors_and_weights(x, y);
    add(a_x, a_y, value * w_a);
    add(b_x, b_y, value * w_b);
    add(c_x, c_y, value * w_c);
    add(d_x, d_y, value * w_d);
  }


  std::array<float, 2> RawImageEigen::compute_bounds() const {
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (auto elem : image.reshaped())
      if (elem != NO_DATA) {
        min_val = std::min(min_val, elem);
        max_val = std::max(max_val, elem);
      }

    // Assert that we have seen at least some valid data.
    assert(max_val != -FLT_MAX);
    assert(min_val != FLT_MAX);

    return {min_val, max_val};
  }


  // multiple times now I've encountered the issue where I can't reduce
  // simple array operations to method calls in Eigen because masking is
  // baked into the image in a way that it doesn't represent the identity
  // of the operation in question. This forces us to unravel the arrays
  // in for loops and check if NO_DATA. I wonder if resorting to Eigens
  // expressions would be faster than this. I also wonder how "dangerous"
  // it would be. How often, let's say, do we expect to see exactly 0 in real data?
  // Then convolution could be as easy as:
  // Image zeroed = (array.array() == NO_DATA).select(0, array.array());
  // result(i, j) = zeroed.block<krows, kcols>(i, j).cwiseProduct(kernel).sum()
  void RawImageEigen::convolve_cpu(PSF& psf) {
    Image result = Image::Zero(height, width);

    const int psf_rad = psf.get_radius();
    const float psf_total = psf.get_sum();

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // Pixels with NO_DATA remain NO_DATA.
        if (image(y, x) == NO_DATA) {
          result(y, x) = NO_DATA;
          continue;
        }

        float sum = 0.0;
        float psf_portion = 0.0;
        for (int j = -psf_rad; j <= psf_rad; j++) {
          for (int i = -psf_rad; i <= psf_rad; i++) {
            if ((x + i >= 0) && (x + i < width) && (y + j >= 0) && (y + j < height)) {
              float current_pixel = image(y+j, x+i);
              if (current_pixel != NO_DATA) {
                float current_psf = psf.get_value(i + psf_rad, j + psf_rad);
                psf_portion += current_psf;
                sum += current_pixel * current_psf;
              }
            }
          } // for i
        } // for j
        result(y, x) = (sum * psf_total) / psf_portion;
      } // for x
    } // for y
    image = std::move(result);
  }

#ifdef HAVE_CUDA
  // Performs convolution between an image represented as an array of floats
  // and a PSF on a GPU device.
  extern "C" void deviceConvolve(float* source_img, float* result_img,
                                 int width, int height, float* psf_kernel,
                                 int psf_size, int psf_dim, int psf_radius,
                                 float psf_sum);
#endif


  void RawImageEigen::convolve(PSF psf) {
#ifdef HAVE_CUDA
    deviceConvolve(image.data(), image.data(),
                   get_width(), get_height(), psf.data(),
                   psf.get_size(), psf.get_dim(), psf.get_radius(), psf.get_sum());
#else
    convolve_cpu(psf);
#endif
  }


  // Imma be honest, this function makes no sense to me... I think I transcribed
  // it right, but I'm not sure? If a flag is in mask, it's masked unless it's in
  // exceptions? But why do we have two lists? Is the example above better than this?
  void RawImageEigen::apply_mask(int flags, const std::vector<int>& exceptions,
                                 const RawImageEigen& mask) {
    const int npixels = get_npixels();
    assert(npixels == mask.get_npixels());

    // evaluated as lvalue because of auto (hopefully)
    auto mask_unravelled = mask.image.reshaped();
    auto image_unravelled = image.reshaped();

    for (unsigned int p = 0; p < npixels; ++p) {
      int pix_flags = static_cast<int>(mask_unravelled(p));
      bool is_exception = false;
      for (auto& e : exceptions) is_exception = is_exception || e == pix_flags;
      if (!is_exception && ((flags & pix_flags) != 0)) image_unravelled(p) = NO_DATA;
    }
  }


  /* This implementation of grow_mask is optimized for steps > 1
     (which is how the code is generally used. If you are only
     growing the mask by 1, the extra copy will be a little slower.
  */
  void RawImageEigen::grow_mask(const int steps) {
    ImageI bitmask = ImageI::Constant(height, width, -1);
    bitmask = (image.array() == NO_DATA).select(0, bitmask);

    for (int itr=1; itr<=steps; ++itr){
      for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i){
          if (bitmask(j, i) == -1){
            if (((j-1 > 0) && (bitmask(j-1, i) == itr-1)) ||
                ((i-1 > 0) && (bitmask(j, i-1) == itr-1)) ||
                ((j+1 < width) && (bitmask(j+1, i) == itr-1)) ||
                ((i+1 < height) && (bitmask(j, i+1) == itr-1))){
              bitmask(j, i) = itr;
            }
          }
        } // for i
      } // for j
    } // for step
    image = (bitmask.array() > -1).select(NO_DATA, image);
  }


  void RawImageEigen::set_all(float value) {
    image.setConstant(value);
  }


  // I kind of honestly forgot "PixelPos" was a class. If we don't need
  // the "furthest from the center" thing the same operation can be
  // replaced by
  // Eigen::Index, i, j;
  // array.maxCoeff(&i, &j)
  // Not sure I understand the condition of !furthest_from_center
  // The maximum value of the image and return the coordinates.
  Index RawImageEigen::find_peak(bool furthest_from_center) const {
    int c_x = width / 2;
    int c_y = height / 2;

    // Initialize the variables for tracking the peak's location.
    Index result = {0, 0};
    float max_val = image(0, 0);
    float dist2 = c_x * c_x + c_y * c_y;

    // Search each pixel for the peak.
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (image(y, x) > max_val) {
          max_val = image(y, x);
          result.i = y;
          result.j = x;
          dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
        }
        else if (image(y, x) == max_val) {
          int new_dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
          if ((furthest_from_center && (new_dist2 > dist2)) ||
              (!furthest_from_center && (new_dist2 < dist2))) {
            max_val = image(y, x);
            result.i = y;
            result.j = x;
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
  ImageMoments RawImageEigen::find_central_moments() const {
    const int num_pixels = width * height;
    const int c_x = width / 2;
    const int c_y = height / 2;

    // Set all the moments to zero initially.
    ImageMoments res = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto pixels = image.reshaped();

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


  // Load the image data from a specific layer of a FITS file.
  void RawImageEigen::load_fits(const std::string& file_path, int layer_num) {
    // Open the file's header and read in the obstime and the dimensions.
    fitsfile* fptr;
    int status = 0;
    int mjdStatus = 0;
    int file_not_found;
    int nullval = 0;
    int anynull = 0;

    // Open the correct layer to extract the RawImageEigen.
    std::string layerPath = file_path + "[" + std::to_string(layer_num) + "]";
    if (fits_open_file(&fptr, layerPath.c_str(), READONLY, &status)) {
      fits_report_error(stderr, status);
      throw std::runtime_error("Could not open FITS file to read RawImageEigen");
    }

    // Read image dimensions.
    long dimensions[2];
    if (fits_read_keys_lng(fptr, "NAXIS", 1, 2, dimensions, &file_not_found, &status))
      fits_report_error(stderr, status);
    width = dimensions[0];
    height = dimensions[1];

    // Read in the image.
    //array = std::vector<float>(width * height);
    image = Image(height, width);
    if (fits_read_img(fptr, TFLOAT, 1, get_npixels(), &nullval, image.data(), &anynull, &status))
      fits_report_error(stderr, status);

    // Read image observation time, ignore error if does not exist
    obstime = -1.0;
    if (fits_read_key(fptr, TDOUBLE, "MJD", &obstime, NULL, &mjdStatus)) obstime = -1.0;
    if (fits_close_file(fptr, &status)) fits_report_error(stderr, status);

    // If we are reading from a sublayer and did not find a time, try the overall header.
    if (obstime < 0.0) {
      if (fits_open_file(&fptr, file_path.c_str(), READONLY, &status))
        throw std::runtime_error("Could not open FITS file to read RawImageEigen");
      fits_read_key(fptr, TDOUBLE, "MJD", &obstime, NULL, &mjdStatus);
      if (fits_close_file(fptr, &status)) fits_report_error(stderr, status);
    }
  }


  void RawImageEigen::save_fits(const std::string& filename) {
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

    // Create the primary array image (32-bit float array)
    long dimensions[2];
    dimensions[0] = width;
    dimensions[1] = height;
    fits_create_img(fptr, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
    fits_report_error(stderr, status);

    /* Write the array of floats to the image */
    fits_write_img(fptr, TFLOAT, 1, get_npixels(), image.data(), &status);
    fits_report_error(stderr, status);

    // Add the basic header data.
    fits_update_key(fptr, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_report_error(stderr, status);

    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
  }


  void RawImageEigen::append_fits_extension(const std::string& filename) {
    int status = 0;
    fitsfile* f;

    // Check that we can open the file.
    if (fits_open_file(&f, filename.c_str(), READWRITE, &status)) {
      fits_report_error(stderr, status);
      throw std::runtime_error("Unable to open FITS file for appending.");
    }

    // This appends a layer (extension) if the file exists)
    /* Create the primary array image (32-bit float array) */
    long dimensions[2];
    dimensions[0] = width;
    dimensions[1] = height;
    fits_create_img(f, FLOAT_IMG, 2 /*naxis*/, dimensions, &status);
    fits_report_error(stderr, status);

    /* Write the array of floats to the image */
    fits_write_img(f, TFLOAT, 1, get_npixels(), image.data(), &status);
    fits_report_error(stderr, status);

    // Save the image time in the header.
    fits_update_key(f, TDOUBLE, "MJD", &obstime, "[d] Generated Image time", &status);
    fits_report_error(stderr, status);

    fits_close_file(f, &status);
    fits_report_error(stderr, status);
  }


  RawImageEigen create_median_image_eigen(const std::vector<RawImageEigen>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) {
      assert(img.get_width() == width and img.get_height() == height);
    }

    RawImageEigen result = RawImageEigen(height, width);

    std::vector<float> pix_array(num_images);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int num_unmasked = 0;
        for (int i = 0; i < num_images; ++i) {
          // Only used the unmasked array.
          float pix_val = images[i].get_image()(y, x);
          if ((pix_val != NO_DATA) && (!std::isnan(pix_val))) { // why are we suddenly checking nans?!
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
            result.get_image()(y, x) = ave_middle;
          } else {
            result.get_image()(y, x) =  pix_array[median_ind];
          }
        } else {
          // We use a 0.0 value if there is no data to allow for visualization
          // and value based filtering.
          result.get_image()(y, x) = 0.0;
        }
      }
    }

    return result;
  }


  RawImageEigen create_summed_image_eigen(const std::vector<RawImageEigen>& images) {
    int num_images = images.size();
    assert(num_images > 0);
    int width = images[0].get_width();
    int height = images[0].get_height();
    RawImageEigen result(height, width);
    for (auto& img : images){
      result.get_image() += (img.get_image().array() == NO_DATA).select(0, img.get_image());
    }
    return result;
  }


  RawImageEigen create_mean_image_eigen(const std::vector<RawImageEigen>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) assert(img.get_width() == width and img.get_height() == height);

    RawImageEigen result = RawImageEigen(height, width);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float sum = 0.0;
        float count = 0.0;
        for (int i = 0; i < num_images; ++i) {
          float pix_val = images[i].get_pixel(y, x);
          // of course...
          if ((pix_val != NO_DATA) && (!std::isnan(pix_val))) {
            count += 1.0;
            sum += pix_val;
          }
        }

        if (count > 0.0) {
          result.set_pixel(y, x, sum / count);
        } else {
          // We use a 0.0 value if there is no data to allow for visualization
          // and value based filtering.
          result.set_pixel(y, x, 0.0);
        }
      } // for x
    } // for y
    return result;
  }


#ifdef Py_PYTHON_H
  static void index_bindings(py::module &m) {
    py::class_<Index>(m, "Index")
      .def(py::init<int, int>())
      .def(py::init<float, float>())
      .def_readwrite("i", &Index::i)
      .def_readwrite("j", &Index::j)
      .def("__iter__", [](const Index &idx){
        std::vector<unsigned> tmp{idx.i, idx.j};
        return py::make_iterator(tmp.begin(), tmp.end());
      }, py::keep_alive<0, 1>())
      .def("__repr__", [] (const Index &idx) { return idx.to_string(); })
      .def("__str__", &Index::to_string);
  }


  static void raw_image_eigen_bindings(py::module &m) {
    using rie = search::RawImageEigen;

    py::class_<rie>(m, "RawImageEigen")
      .def(py::init<>())
      .def(py::init<search::RawImageEigen&>())
      .def(py::init<search::Image&, double>(),
           py::arg("img").noconvert(true), py::arg("obs_time")=-1.0d)
      .def(py::init<unsigned, unsigned, float, double>(),
           py::arg("w"), py::arg("h"), py::arg("value")=0.0f, py::arg("obs_time")=-1.0d)
      .def_property_readonly("height", &rie::get_height)
      .def_property_readonly("width", &rie::get_width)
      .def_property_readonly("npixels", &rie::get_npixels)
      .def_property("obstime", &rie::get_obstime, &rie::set_obstime)
      .def_property("image", py::overload_cast<>(&rie::get_image, py::const_), &rie::set_image)
      .def_property("imref", py::overload_cast<>(&rie::get_image), &rie::set_image)
      .def("get_pixel", &rie::get_pixel)
      .def("pixel_has_data", &rie::pixel_has_data)
      .def("set_pixel", &rie::set_pixel)
      .def("set_all", &rie::set_all)
      .def("l2_allclose", &rie::l2_allclose)
      .def("compute_bounds", &rie::compute_bounds)
      .def("find_peak", &rie::find_peak)
      .def("find_central_moments", &rie::find_central_moments)
      .def("create_stamp", &rie::create_stamp, py::return_value_policy::copy)
      .def("interpolate", &rie::interpolate)
      .def("interpolated_add", &rie::interpolated_add)
      .def("apply_mask", &rie::apply_mask)
      .def("grow_mask", &rie::grow_mask)
      .def("convolve", &rie::convolve)
      .def("convolve_cpu", &rie::convolve_cpu)
      .def("save_fits", &rie::save_fits)
      .def("append_fits_extension", &rie::append_fits_extension)
      .def("load_fits", &rie::load_fits);
  }
#endif

} /* namespace search */
