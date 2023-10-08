#include "raw_image_eigen.h"


namespace py = pybind11;


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


  RawImageEigen::RawImageEigen(unsigned w, unsigned h, float value, double obs_time)
    : height(h),
      width(w),
      obstime(obs_time) {
    if (value != 0)
      image = Image::Constant(height, width, value);
    else
      image = Image::Zero(height, width);
  }


  // Copy constructor
  RawImageEigen::RawImageEigen(const RawImageEigen& old) {
    width = old.get_width();
    height = old.get_height();
    image = old.get_array();
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


  bool RawImageEigen::isclose(const RawImageEigen& img_b, float atol) const {
    return image.isApprox(img_b.image, atol);
  }


  auto RawImageEigen::get_interp_neighbors_and_weights(const float x, const float y) const {
    Point p(x, y);

    // coordinates of nearest neighboring array
    // top left (tl), top right (tr),
    // bottom left (bl), bottom right (br)
    auto [tl, tr, bl, br] = p.nearest_pixel_coords();

    // interpolation weights for each of the 4 neighbors
    // total normalization should be 1 always (size of array)
    // float normalization = 1 / ((tr.x - tl.x) * (tl.y - bl.y));
    float w_tl = (tr.x - x) * (y - bl.y);
    float w_tr = (x - tl.x) * (y - bl.y);
    float w_bl = (br.x - x) * (tr.y - y);
    float w_br = (x - bl.x) * (tl.y - y);

    auto [itl, itr, ibl, ibr] = p.nearest_pixel_idxs();

    struct retval {
      // gee golly naming is hard...
      Index top_left, top_right, bot_left, bot_right;
      float w_tl, w_tr, w_bl, w_br;
    };

    return retval{itl, itr, ibl, ibr, w_tl, w_tr, w_bl, w_br};
  }


  float RawImageEigen::interpolate(const float x, const float y) const {
    Point p(x, y);

    // neighbors and weights for interpolation
    auto iwn = get_interp_neighbors_and_weights(x, y);

    // I sure hope the compiler optimizes locals away successfully
    auto tl = iwn.top_left;
    auto tr = iwn.top_right;
    auto bl = iwn.bot_left;
    auto br = iwn.bot_right;

    // dodge masked values (could also have been an expression,
    // or for loop, not sure about preferred C++ ways or benefits?
    //return (w_tl * (arr(tl.x, tl.y) == NO_DATA) ? 0.0f, arr(tl.x, tl.y) +
    //        w_tr * (arr(tr.x, tr.y) == NO_DATA) ? 0.0f, arr(tl.x, tl.y) +
    //        w_bl * (arr(bl.x, bl.y) == NO_DATA) ? 0.0f, arr(tl.x, tl.y) +
    //        w_br * (arr(br.x, br.y) == NO_DATA) ? 0.0f, arr(tl.x, tl.y));
    float interpolated_value = 0.0f;
    float sum_weights = 0.0f;
    if (image(tl.i, tl.j) != NO_DATA){
      sum_weights += iwn.w_tl;
      interpolated_value += iwn.w_tl * image(tl.i, tl.j);
    }

    if (image(tr.i, tr.j) != NO_DATA){
      sum_weights += iwn.w_tl;
     interpolated_value += iwn.w_tr * image(tr.i, tr.j);
     }

    if (image(bl.i, bl.j) != NO_DATA) {
      sum_weights += iwn.w_tl;
      interpolated_value += iwn.w_bl * image(bl.i, bl.j);
    }

    if (image(br.i, br.j) != NO_DATA) {
      sum_weights += iwn.w_tl;
      interpolated_value += iwn.w_br * image(br.i, br.j);
    }

    if (sum_weights == 0.0f)
      return NO_DATA;
    return interpolated_value/sum_weights;
  }

  RawImageEigen RawImageEigen::create_stamp(const float x, const float y, const int radius,
                                            const bool interpolate, const bool keep_no_data) const {
    if (radius < 0)
      throw std::runtime_error("stamp radius must be at least 0");

    const int dim = radius * 2 + 1;
    Index idx(x, y, false);
    // evaluates as rvalue. The constructor assigns the block to an array, type
    // Image, prompting the evaluation of the expression
    // to fix the constness issue we will materialize a copy of the array
    auto [i, dimx, j, dimj] = idx.center_block(dim, width, height)
    Image block = image.block(i, j, dimx, dimy);
    RawImageEigen stamp(block, obstime);

    if (interpolate) {
      for (int yoff = 0; yoff < dim; ++yoff) {
        for (int xoff = 0; xoff < dim; ++xoff) {
          stamp.image(yoff, xoff) = this->interpolate(x + static_cast<float>(xoff - radius),
                                                      y + static_cast<float>(yoff - radius));
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
    auto iwn = get_interp_neighbors_and_weights(x, y);
    add(iwn.top_left,  value * iwn.w_tl);
    add(iwn.top_right, value * iwn.w_tr);
    add(iwn.bot_left,  value * iwn.w_bl);
    add(iwn.bot_right, value * iwn.w_br);
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

    // cast the PSF kernel as an Eigen Matrix for cleanliness
    // remember to undo later. Also, what is the PSF dimension?
    Eigen::Map<Eigen::MatrixXf> kernel_raw(psf.data(), 1, psf.get_size());
    auto kernel = kernel_raw.reshaped(psf.get_dim(), psf.get_dim());

    const int psf_rad = psf.get_radius();
    const float psf_total = psf.get_sum();

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // Pixels with NO_DATA remain NO_DATA.
        if (image(x, y) == NO_DATA){
          result(x, y) = NO_DATA;
          continue;
        }

        float sum = 0.0;
        float psf_portion = 0.0;
        for (int j = -psf_rad; j <= psf_rad; j++) {
          for (int i = -psf_rad; i <= psf_rad; i++) {
            Index idx(x+i, y+j);
            if (contains(idx) &&  image(idx.i, idx.j) != NO_DATA){
              psf_portion += kernel(i+psf_rad, j+psf_rad);
              sum += image(idx.i, idx.j) * kernel(i+psf_rad, j+psf_rad);
            }
          } // for i
        } // for j
        result(x, y) = (sum * psf_total) / psf_portion;
      } // for x
    } // for y

    // avoid copying the results back
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


  void RawImageEigen::apply_bitmask(const ImageI& bitmask) {
    image = (bitmask.array() == 1).select(NO_DATA, image);
  }


  void RawImageEigen::apply_mask1(const ImageI& mask, const int flags) {
    // I'm not sure a copy is avoidable here, we can pass a non-const Ref
    // and then edit the mask in-place, but that would edit the Python-side
    // numpy array as well....
    auto bitmask = mask.unaryExpr( [&flags](int x){return flags & x;} );
    apply_bitmask(bitmask);
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
  void RawImageEigen::grow_mask(const unsigned steps) {
    const int num_array = width * height;

    // Set up the initial masked vector that stores the number of steps
    // each pixel is from a masked pixel in the original image.
    ImageI bitmask = ImageI::Zero(height, width);
    bitmask = (image.array() == NO_DATA).select(1, bitmask).reshaped();

    // Grow out the mask one for each step.
    for (int itr = 1; itr <= steps; ++itr) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int center = width * y + x;
          if (bitmask[center] == -1) {
            // Mask pixels that are adjacent to a pixel masked during
            // the last iteration only.
            if ((x + 1 < width && bitmasked[center + 1] == itr - 1) ||
                (x - 1 >= 0 && bitmasked[center - 1] == itr - 1) ||
                (y + 1 < height && bitmasked[center + width] == itr - 1) ||
                (y - 1 >= 0 && bitmasked[center - width] == itr - 1)) {
              bitmasked[center] = itr;
            }
          }
        }
      }
    }

    apply_bitmask(bitmask);
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
        if (image(x, y) > max_val) {
          max_val = image(x, y);
          result.i = x;
          result.j = y;
          dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
        }
        else if (image(x, y) == max_val) {
          int new_dist2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y);
          if ((furthest_from_center && (new_dist2 > dist2)) ||
              (!furthest_from_center && (new_dist2 < dist2))) {
            max_val = image(x, y);
            result.i = x;
            result.j = y;
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
    // Set all the moments to zero initially.
    ImageMoments res = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Find the min (non-NO_DATA) value to subtract off.
    auto min_val = (image.array() != NO_DATA).select(9999.9999f, image).minCoeff();

    // find the sum of the zero-shifted non-NO_DATA array
    // first offset the array by the minimum then select, from original array,
    // all the values of NO_DATA as zero and then finally select remaining
    // values from the shifted array. Note that everything evaluates as an
    // expression, waiting and not materializing a matrix in memory until we
    // cast it to one. I've been looking at godbolt assembly trying to figure out
    // if it's better to leave normed as an expression until division or not, but
    // it's too clever for me. I think it uses some vectorized commands, but I've
    // never seen them before...
    auto tmp = image.array() - min_val;
    Image normed = (image.array() == NO_DATA).select(0, tmp);
    auto sum = normed.sum();
    if (sum == 0.0){
      return res;
    }

    // If the image isn't empty, compute the rest of the moments.
    normed /= normed.sum();
    const unsigned c_x = width / 2;
    const unsigned c_y = height / 2;
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        res.m00 += normed(x, y);
        res.m10 += (x - c_x) * normed(x, y);
        res.m20 += (x - c_x) * (x - c_x) * normed(x, y);
        res.m01 += (y - c_y) * normed(x, y);
        res.m02 += (y - c_y) * (y - c_y) * normed(x, y);
        res.m11 += (x - c_x) * (y - c_y) * normed(x, y);
      }
    }
    return res;
  }

    // Load the image data from a specific layer of a FITS file.
  void RawImageEigen::load_from_file(const std::string& file_path, int layer_num) {
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


  void RawImageEigen::save_to_file(const std::string& filename) {
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


  void RawImageEigen::append_layer_to_file(const std::string& filename) {
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


  // my implementation here, to test performance for this because no way the
  // optimization from the original one actually can be worth the messiness
  const float calc_median(std::vector<float>& values) {
    std::sort(values.begin(), values.end());
    const auto len = values.size();
    return len % 2 == 0 ?
      (values[len/2-1] + values[len/2]) / 2.0f :
      values[len/2];
  }


  RawImageEigen create_median_image_eigen(const std::vector<RawImageEigen>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    int width = images[0].get_width();
    int height = images[0].get_height();
    for (auto& img : images) {
      assert(img.get_width() == width and img.get_height() == height);
    }

    RawImageEigen result = RawImageEigen(width, height);

    std::vector<float> pix_array(num_images);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int num_unmasked = 0;
        for (int i = 0; i < num_images; ++i) {
          // Only used the unmasked array.
          float pix_val = images[i].get_pixel(x, y);
          if ((pix_val != NO_DATA) && (!std::isnan(pix_val))) { // why are we suddenly checking nans?!
            pix_array[num_unmasked] = pix_val;
            num_unmasked += 1;
          }
        }

        if (num_unmasked > 0) {
          // how is it possible that this optimization ever pays off?
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


  RawImageEigen create_summed_image_eigen(const std::vector<RawImageEigen>& images) {
    int num_images = images.size();
    assert(num_images > 0);
    int width = images[0].get_width();
    int height = images[0].get_height();
    RawImageEigen result(width, height);
    for (auto& img : images){
      result.image += img.image;
    }
    return result;
  }


  RawImageEigen create_mean_image_eigen(const std::vector<RawImageEigen>& images) {
    int num_images = images.size();
    assert(num_images > 0);

    auto result = create_summed_image_eigen(images);
    result.image.array() /= num_images;
    result.image = (result.image.array() == NO_DATA).select(0, result.image);
    return result;
  }


#ifdef Py_PYTHON_H
  static void raw_image_eigen_bindings(py::module &m) {
    using rie = search::RawImageEigen;

    py::class_<rie>(m, "RawImageEigen")
      .def(py::init<>())
      .def(py::init<search::Image&, double>(),
           py::arg("img").noconvert(true), py::arg("obstime"))
      .def(py::init<unsigned, unsigned, float, double>(),
           py::arg("w"), py::arg("h"), py::arg("value")=0.0f, py::arg("obs_time")=-1.0d)
      .def("height", &rie::get_height)
      .def("width", &rie::get_width)
      //.def_property("data",
      //              /*getter=*/ [](rie& cls) { return ImageRef(cls.data); },
      //              /*setter=*/ [](rie& cls, ImageRef value) { cls.array = value; })
      .def_property("obstime", &rie::get_obstime, &rie::set_obstime)
      .def("isclose", &rie::isclose)
      .def("compute_bounds", &rie::compute_bounds)
      .def("find_peak", &rie::find_peak)
      .def("find_central_moments", &rie::find_central_moments)
      .def("create_stamp", &rie::create_stamp)
      .def("interpolate", &rie::interpolate)
      .def("interpolated_add", &rie::interpolated_add)
      // I really need to confirm whether or not references make a copy
      // or behave like Eigen::Ref counterparts - we might want to use more of
      // those in our method signatures then
      .def("apply_bitmask", &rie::apply_bitmask)
      .def("apply_mask1", &rie::apply_mask1)
      .def("apply_mask", &rie::apply_mask)
      //.def("apply_mask", py::overload_cast<search::ImageI&, int>(&rie::apply_mask))
      //.def("apply_mask", py::overload_cast<int, std::vector<int>&, rie&>(&rie::apply_mask));
      .def("grow_mask", &rie::grow_mask)
      .def("pixel_has_data", &rie::pixel_has_data)
      .def("convolve", &rie::convolve)
      .def("convolve_cpu", &rie::convolve_cpu);
  }
#endif

} /* namespace search */
