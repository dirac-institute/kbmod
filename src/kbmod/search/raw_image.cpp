#include "raw_image.h"


namespace search {
  using Index = indexing::Index;
  using Point = indexing::Point;

  RawImage::RawImage()
    : width(0),
      height(0),
      obstime(-1.0),
      image() {}


  RawImage::RawImage(Image& img, double obs_time)  {
    image = std::move(img);
    height = image.rows();
    width = image.cols();
    obstime = obs_time;
  }


  RawImage::RawImage(unsigned h, unsigned w, float value, double obs_time)
    : height(h),
      width(w),
      obstime(obs_time) {
    if (value != 0.0f)
      image = Image::Constant(height, width, value);
    else
      image = Image::Zero(height, width);
  }


// Copy constructor
RawImage::RawImage(const RawImage& old) {
    width = old.get_width();
    height = old.get_height();
    image = old.get_image();
    obstime = old.get_obstime();
}


  // Move constructor
  RawImage::RawImage(RawImage&& source)
    : width(source.width),
      height(source.height),
      obstime(source.obstime),
      image(std::move(source.image)) {}


  // Copy assignment
  RawImage& RawImage::operator=(const RawImage& source) {
    width = source.width;
    height = source.height;
    image = source.image;
    obstime = source.obstime;
    return *this;
}


// Move assignment
RawImage& RawImage::operator=(RawImage&& source) {
    if (this != &source) {
      width = source.width;
      height = source.height;
      image = std::move(source.image);
      obstime = source.obstime;
    }
    return *this;
}


  bool RawImage::l2_allclose(const RawImage& img_b, float atol) const {
    // https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
    return image.isApprox(img_b.image, atol);
  }


  inline auto RawImage::get_interp_neighbors_and_weights(const Point& p) const {
    // top/bot left; bot/top right
    auto neighbors = indexing::manhattan_neighbors(p, width, height);

    float tmp_integral;
    float x_frac = std::modf(p.x-0.5f, &tmp_integral);
    float y_frac = std::modf(p.y-0.5f, &tmp_integral);

    // weights for top/bot left; bot/top right
    std::array<float, 4> weights{
      (1-x_frac) * (1-y_frac),
      (1-x_frac) *    y_frac,
      x_frac  *    y_frac,
      x_frac  * (1-y_frac)
    };

    struct NeighborsAndWeights{
      std::array<std::optional<Index>, 4> neighbors;
      std::array<float, 4> weights;
    };

    return NeighborsAndWeights{neighbors, weights};
  }


  float RawImage::interpolate(const Point& p) const {
    if (!contains(p))
      return NO_DATA;

    auto [neighbors, weights] = get_interp_neighbors_and_weights(p);

    // this is the way apparently the way it's supposed to be done
    // too bad the loop couldn't have been like
    // for (auto& [neighbor, weight] : std::views::zip(neigbors, weights))
    // https://en.cppreference.com/w/cpp/ranges/zip_view
    // https://en.cppreference.com/w/cpp/utility/optional
    float sumWeights = 0.0;
    float total = 0.0;
    for (int i=0; i<neighbors.size(); i++){
      if (auto idx = neighbors[i]){
        if (pixel_has_data(*idx)){
          sumWeights += weights[i];
          total += weights[i] * image(idx->j, idx->i);
        }
      }
    }

    if (sumWeights == 0.0)
      return NO_DATA;
    return total / sumWeights;
  }


  RawImage RawImage::create_stamp(const Point& p,
                                  const int radius,
                                  const bool interpolate,
                                  const bool keep_no_data) const {
    if (radius < 0)
      throw std::runtime_error("stamp radius must be at least 0");

    const int dim = radius*2 + 1;
    // can't address this instance of non-uniform index handling with Point
    // and Index, because at a base level it adopts a different definition of
    // the pixel grid to coordinate system transformation.
    auto [corner, anchor, h, w] = indexing::anchored_block({(int)p.y, (int)p.x}, radius,
                                                           width, height);

    Image stamp;
    if (keep_no_data)
      stamp = Image::Constant(dim, dim, NO_DATA);
    else
      stamp = Image::Zero(dim, dim);
    stamp.block(anchor.j, anchor.i, h, w) = image.block(corner.j, corner.i, h, w);

    if (interpolate) {
      for (int yoff = 0; yoff < dim; ++yoff) {
        for (int xoff = 0; xoff < dim; ++xoff) {
          // I think the {} create a temporary, but I don't know how bad that is
          // would it be the same if we had interpolate just take 2 floats?
          stamp(yoff, xoff) = this->interpolate({
              p.y + static_cast<float>(yoff - radius),
              p.x + static_cast<float>(xoff - radius)
            });
        }
      }
    }
    return RawImage(stamp);
  }


  inline void RawImage::add(const Index& idx, const float value) {
    if (contains(idx))
      image(idx.j, idx.i) += value;
  }


  inline void RawImage::add(const Point& p, const float value) {
    add(p.to_index(), value);
  }


  void RawImage::interpolated_add(const Point& p, const float value) {
    auto [neighbors, weights] = get_interp_neighbors_and_weights(p);
    for (int i=0; i<neighbors.size(); i++){
      if (auto& idx = neighbors[i]){
        add(*idx, value*weights[i]);
      }
    }
  }


  std::array<float, 2> RawImage::compute_bounds() const {
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


  void RawImage::convolve_cpu(PSF& psf) {
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
              // note that convention for index access is flipped for PSF
              if (current_pixel != NO_DATA) {
                float current_psf = psf.get_value(i + psf_rad, j + psf_rad);
                psf_portion += current_psf;
                sum += current_pixel * current_psf;
              }
            }
          } // for i
        } // for j
        if (psf_portion == 0){
          result(y, x) = NO_DATA;
        } else {
          result(y, x) = (sum * psf_total) / psf_portion;
        }
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


  void RawImage::convolve(PSF psf) {
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
  void RawImage::apply_mask(int flags, const std::vector<int>& exceptions,
                                 const RawImage& mask) {
    for (unsigned int j=0; j<height; ++j){
      for (unsigned int i=0; i<height; ++i){
        int pix_flags = static_cast<int>(mask.image(j, i));
        bool is_exception = false;
        for (auto& e : exceptions){
          is_exception = is_exception || e == pix_flags;
        }
        if (!is_exception && ((flags & pix_flags) != 0)) {
          image(j, i) = NO_DATA;
        }
      } // for i
    } // for j
  }


  /* This implementation of grow_mask is optimized for steps > 1
     (which is how the code is generally used. If you are only
     growing the mask by 1, the extra copy will be a little slower.
  */
  void RawImage::grow_mask(const int steps) {
    ImageI bitmask = ImageI::Constant(height, width, -1);
    bitmask = (image.array() == NO_DATA).select(0, bitmask);

    for (int itr=1; itr<=steps; ++itr){
      for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i){
          if (bitmask(j, i) == -1){
            if (((j-1 > 0) && (bitmask(j-1, i) == itr-1)) ||
                ((i-1 > 0) && (bitmask(j, i-1) == itr-1)) ||
                ((j+1 < height) && (bitmask(j+1, i) == itr-1)) ||
                ((i+1 < width) && (bitmask(j, i+1) == itr-1))){
              bitmask(j, i) = itr;
            }
          }
        } // for i
      } // for j
    } // for step
    image = (bitmask.array() > -1).select(NO_DATA, image);
  }


  void RawImage::set_all(float value) {
    image.setConstant(value);
  }


  // I kind of honestly forgot "PixelPos" was a class. If we don't need
  // the "furthest from the center" thing the same operation can be
  // replaced by
  // Eigen::Index, i, j;
  // array.maxCoeff(&i, &j)
  // Not sure I understand the condition of !furthest_from_center
  // The maximum value of the image and return the coordinates.
  Index RawImage::find_peak(bool furthest_from_center) const {
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


  void RawImage::load_time_from_file(fitsfile* fptr) {
    int mjd_status = 0;

    // Set object's obstime directly to -1 to indicate no time found.
    obstime = -1.0;

    // Read image observation time, trying the MJD field first and DATE-AVG second.
    // Ignore error if does not exist.
    if (fits_read_key(fptr, TDOUBLE, "MJD", &obstime, NULL, &mjd_status)) {

      // Reset the status flag and try with DATE-AVG.
      mjd_status = 0;
      if (fits_read_key(fptr, TDOUBLE, "DATE-AVG", &obstime, NULL, &mjd_status)) {
        obstime = -1.0;
      }
    }
  }


  // Load the image data from a specific layer of a FITS file.
  void RawImage::from_fits(const std::string& file_path, int layer_num) {
    // Open the file's header and read in the obstime and the dimensions.
    fitsfile* fptr;
    int status = 0;
    int mjdStatus = 0;
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
    //array = std::vector<float>(width * height);
    image = Image(height, width);
    if (fits_read_img(fptr, TFLOAT, 1, get_npixels(), &nullval, image.data(), &anynull, &status))
      fits_report_error(stderr, status);

    // Read image observation time, ignore error if does not exist
    load_time_from_file(fptr);
    if (fits_close_file(fptr, &status)) fits_report_error(stderr, status);

    // If we are reading from a sublayer and did not find a time, try the overall header.
    if (obstime < 0.0) {
      if (fits_open_file(&fptr, file_path.c_str(), READONLY, &status))
        throw std::runtime_error("Could not open FITS file to read RawImage");
      fits_read_key(fptr, TDOUBLE, "MJD", &obstime, NULL, &mjdStatus);
      if (fits_close_file(fptr, &status)) fits_report_error(stderr, status);
    }
  }


  void RawImage::to_fits(const std::string& filename) {
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


  void RawImage::append_to_fits(const std::string& filename) {
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


  // it makes no sense to return RawImage here because there is no
  // obstime by definition of operation, but I guess it's out of
  // scope for this PR because it requires updating layered_image
  // and image stack
  RawImage create_median_image(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) {
      assert(img.cols() == width and img.rows() == height);
    }

    Image result = Image::Zero(height, width);

    std::vector<float> pix_array(num_images);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int num_unmasked = 0;
        for (auto& img : images) {
          // Only used the unmasked array.
          // we have a get_pixel and pixel_has_data, but we don't use them
          // here in the original code, so I go to get_image()() too...
          if ((img.get_image()(y, x) != NO_DATA) && (!std::isnan(img.get_image()(y, x)))) { // why are we suddenly checking nans?!
            pix_array[num_unmasked] = img.get_image()(y, x);
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
            result(y, x) = ave_middle;
          } else {
            result(y, x) =  pix_array[median_ind];
          }
        } else {
          // We use a 0.0 value if there is no data to allow for visualization
          // and value based filtering.
          result(y, x) = 0.0;
        }
    }

    return RawImage(result);
  }


  RawImage create_summed_image(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) assert(img.get_width() == width and img.get_height() == height);

    Image result = Image::Zero(height, width);
    for (auto& img : images){
      result += (img.get_image().array() == NO_DATA).select(0, img.get_image());
    }
    return RawImage(result);
  }


 RawImage create_mean_image(const std::vector<RawImage>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images)
      assertm(img.get_width() == width and img.get_height() == height,
              "Can not sum images with different dimensions.");

    Image result = Image::Zero(height, width);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float sum = 0.0;
        float count = 0.0;
        for (auto& img : images) {
          // we have a get_pixel and pixel_has_data, but we don't use them
          // here in the original code, so I go to get_image()() too...
          if ((img.get_image()(y, x) != NO_DATA) && (!std::isnan(img.get_image()(y, x)))) {
            count += 1.0;
            sum += img.get_image()(y, x);
          }
        }

        if (count > 0.0)
          result(y, x) = sum/count;
        else
          result(y, x) = 0.0; // use 0 for visualization purposes
      } // for x
    } // for y
    return RawImage(result);
  }


#ifdef Py_PYTHON_H
  static void raw_image_bindings(py::module &m) {
    using rie = search::RawImage;

    py::class_<rie>(m, "RawImage")
      .def(py::init<>())
      .def(py::init<search::RawImage&>())
      .def(py::init<search::Image&, double>(),
           py::arg("img").noconvert(true), py::arg("obs_time")=-1.0d)
      .def(py::init<unsigned, unsigned, float, double>(),
           py::arg("w"), py::arg("h"), py::arg("value")=0.0f, py::arg("obs_time")=-1.0d)
      // attributes and properties
      .def_property_readonly("height", &rie::get_height)
      .def_property_readonly("width", &rie::get_width)
      .def_property_readonly("npixels", &rie::get_npixels)
      .def_property("obstime", &rie::get_obstime, &rie::set_obstime)
      .def_property("image", py::overload_cast<>(&rie::get_image, py::const_), &rie::set_image)
      .def_property("imref", py::overload_cast<>(&rie::get_image), &rie::set_image)
      // pixel accessors and setters
      .def("get_pixel", &rie::get_pixel)
      .def("pixel_has_data", &rie::pixel_has_data)
      .def("set_pixel", &rie::set_pixel)
      .def("set_all", &rie::set_all)
      // python interface adapters (avoids having to construct Index & Point)
      .def("get_pixel", [](rie& cls, int j, int i){
        return cls.get_pixel({j, i});
      })
      .def("pixel_has_data", [](rie& cls, int j, int i){
        return cls.pixel_has_data({j, i});
      })
      .def("set_pixel", [](rie& cls, int j, int i, double val){
        cls.set_pixel({j, i}, val);
      })
      // methods
      .def("l2_allclose", &rie::l2_allclose)
      .def("compute_bounds", &rie::compute_bounds)
      .def("find_peak", &rie::find_peak)
      .def("find_central_moments", &rie::find_central_moments)
      .def("create_stamp", &rie::create_stamp)
      .def("interpolate", &rie::interpolate)
      .def("interpolated_add", &rie::interpolated_add)
      .def("get_interp_neighbors_and_weights", [](rie& cls, float y, float x){
        auto tmp = cls.get_interp_neighbors_and_weights({y, x});
        return (tmp.neighbors, tmp.weights);
          })
      .def("apply_mask", &rie::apply_mask)
      .def("grow_mask", &rie::grow_mask)
      .def("convolve_gpu", &rie::convolve)
      .def("convolve_cpu", &rie::convolve_cpu)
      .def("save_fits", &rie::to_fits)
      .def("append_fits_extension", &rie::append_to_fits)
      .def("load_fits", &rie::from_fits)
      // python interface adapters
      .def("create_stamp", [](rie& cls, float y, float x, int radius,
                              bool interp, bool keep_no_data){
        return cls.create_stamp({y, x}, radius, interp, keep_no_data );
      })
      .def("interpolate", [](rie& cls, float y, float x){
        return cls.interpolate({y, x});
      })
      .def("interpolated_add", [](rie& cls, float y, float x, float val){
        cls.interpolated_add({y, x}, val);
      });
  }
#endif

} /* namespace search */
