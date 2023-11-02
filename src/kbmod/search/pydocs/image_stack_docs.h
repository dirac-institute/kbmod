#ifndef IMAGESTACK_DOCS
#define IMAGESTACK_DOCS

namespace pydocs {
static const auto DOC_ImageStack = R"doc(
  A class for storing a list of LayeredImages at different times.
  )doc";

static const auto DOC_ImageStack_get_images = R"doc(
  Returns a reference to the vector of images.
  )doc";

static const auto DOC_ImageStack_img_count = R"doc(
  Returns the number of images in the stack.
  )doc";

static const auto DOC_ImageStack_get_single_image = R"doc(
  Returns a single LayeredImage for a given index.
  )doc";

static const auto DOC_ImageStack_get_obstime = R"doc(
  Returns a single image's observation time in MJD.
  )doc";

static const auto DOC_ImageStack_get_zeroed_time = R"doc(
  Returns a single image's observation time relative to that
  of the first image.
  )doc";

static const auto DOC_ImageStack_build_zeroed_times = R"doc(
  Construct an array of time differentials between each image
  in the stack and the first image.
  ")doc";

static const auto DOC_ImageStack_apply_mask_flags = R"doc(
  Applies a mask to each image by comparing the given bit vector with the
  values in the mask layer and marking pixels NO_DATA. Modifies the image in-place.

  Parameters
  ----------
  flag : `int`
      The bit mask of mask flags to use.
  exceptions : `list` of `int`
      A list of exceptions (combinations of bits where we do not apply the mask).
  )doc";

static const auto DOC_ImageStack_apply_mask_threshold = R"doc(
  Applies a threshold mask to each image by setting pixel values over
  a given threshold to NO_DATA. Modifies the images in-place.

  Parameters
  ----------
  thresh : `float`
      The threshold value to use.
  )doc";

static const auto DOC_ImageStack_apply_global_mask = R"doc(
  Createas a global mask an applies it to each image. A global mask
  masks a pixel if and only if that pixel is masked in at least ``threshold``
  individual images.  Modifies the images in-place and creates the global mask.

  Parameters
  ----------
  flag : `int`
      The bit mask of mask flags to use.
  threshold : `int`
      The minimum number of images in which a pixel must be masked to be
      part of the global mask.
  )doc";

static const auto DOC_ImageStack_grow_mask = R"doc(
  Expands the NO_DATA tags to nearby pixels for all images.
  Modifies the images in-place.

  Parameters
  ----------
  steps : `int`
     The number of pixels by which to grow the masked regions.
  )doc";

static const auto DOC_ImageStack_save_global_mask = R"doc(
  Saves the global mask created by apply_global_mask to a FITS file.

  Parameters
  ----------
  path : `str`
      The directory in which to store the global mask file.
  )doc";

static const auto DOC_ImageStack_save_images = R"doc(
  Saves each image in the stack to its own FITS file.

  Saves the file as {path}/{filename}.fits where the path is given
  and the file name is an object attribute.

  Parameters
  ----------
  path : `str`
      The file path to use. 
  )doc";

static const auto DOC_ImageStack_get_global_mask = R"doc(
  Returns a reference to the global mask created by apply_global_mask.
  )doc";

static const auto DOC_ImageStack_convolve_psf = R"doc(
  Convolves each image (science and variance layers) with the PSF
  stored in the LayeredImage object.
  )doc";

static const auto DOC_ImageStack_get_width = R"doc(
  Returns the width of the images in pixels.
  )doc";

static const auto DOC_ImageStack_get_height = R"doc(
  Returns the height of the images in pixels.
  )doc";

static const auto DOC_ImageStack_get_npixels = R"doc(
  Returns the number of pixels per image.
  )doc";

}  // namespace pydocs

#endif /* IMAGESTACK_DOCS */
