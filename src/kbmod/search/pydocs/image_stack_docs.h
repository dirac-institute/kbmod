#ifndef IMAGESTACK_DOCS
#define IMAGESTACK_DOCS

namespace pydocs {
static const auto DOC_ImageStack = R"doc(
  A class for storing a list of LayeredImages at different times.
      
  Notes
  -----
  The images are not required to be in sorted time order, but the first
  image is used for t=0.0 when computing zeroed times (which might make
  some times negative).
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
  of the first image. This can return negative times if the
  images are not sorted by time.
  )doc";

static const auto DOC_ImageStack_build_zeroed_times = R"doc(
  Construct an array of time differentials between each image
  in the stack and the first image. This can return negative times
  if the images are not sorted by time.
  ")doc";

static const auto DOC_ImageStack_sort_by_time = R"doc(
  Sort the images in the ImageStack by their time.
  ")doc";

static const auto DOC_ImageStack_make_global_mask = R"doc(
  Create a new global mask from a set of flags and a threshold.
  The global mask marks a pixel as masked if and only if it is masked
  by one of the given flags in at least ``threshold`` individual images.
  The returned mask is binary.

  Parameters
  ----------
  flags : `int`
      A bit mask of mask flags to use when counting.
  threshold : `int`
      The minimum number of images in which a pixel must be masked to be
      part of the global mask.

  Returns
  -------
  global_mask : `RawImage`
      A RawImage containing the global mask with 1 for masked pixels
      and 0 for unmasked pixels.
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
