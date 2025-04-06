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

static const auto DOC_ImageStack_on_gpu = R"doc(
  Indicates whether a copy of the images are stored on GPU.
   
  Returns
  -------
  on_gpu : `bool`
      Indicates whether a copy of the images are stored on GPU.
  )doc";

static const auto DOC_ImageStack_get_images = R"doc(
  Returns a reference to the vector of LayeredImages.
   
  Returns
  -------
  images : `list`
      The reference to the vector of LayeredImages.
  )doc";

static const auto DOC_ImageStack_img_count = R"doc(
  Returns the number of images in the stack.

  Returns
  -------
  img_count : `int`
      The number of images in the stack.
  )doc";

static const auto DOC_ImageStack_get_single_image = R"doc(
  Returns a single LayeredImage for a given index.

  Parameters
  ----------
  index : `int`
      The index of the LayeredImage to retrieve.

  Returns
  -------
  `LayeredImage`

  Raises
  ------
  Raises a ``IndexError`` if the index is out of bounds.
)doc";

static const auto DOC_ImageStack_set_single_image = R"doc(
  Sets a single LayeredImage for at a given index.

  Parameters
  ----------
  index : `int`
      The index of the LayeredImage to set.
  img : `LayeredImage`
      The new image.
  force_move : `bool`
      Use move semantics. The input layered image is destroyed to avoid
      a copy of the LayeredImage.

  Raises
  ------
  Raises a ``IndexError`` if the index is out of bounds.
  Raises a ``RuntimeError`` if the input image is the wrong size or the data
  is currently on GPU.
  )doc";

static const auto DOC_ImageStack_append_image = R"doc(
  Appends a single LayeredImage to the back of the ImageStack.

  Parameters
  ----------
  img : `LayeredImage`
      The new image.
  force_move : `bool`
      Use move semantics. The input layered image is destroyed to avoid
      a copy of the LayeredImage.

  Raises
  ------
  Raises a ``RuntimeError`` if the input image is the wrong size or the data
  is currently on GPU.
  )doc";

static const auto DOC_ImageStack_get_obstime = R"doc(
  Returns a single image's observation time in MJD.

  Parameters
  ----------
  index : `int`
      The index of the LayeredImage to retrieve.

  Returns
  -------
  time : `double`
      The observation time (in UTC MJD).

  Raises
  ------
  Raises a ``IndexError`` if the index is out of bounds.
  )doc";

static const auto DOC_ImageStack_get_zeroed_time = R"doc(
  Returns a single image's observation time relative to that
  of the first image. This can return negative times if the
  images are not sorted by time.

  zeroed_time[i] = time[i] - time[0] 

  Parameters
  ----------
  index : `int`
      The index of the LayeredImage to retrieve.

  Returns
  -------
  time : `double`
      The zeroed observation time (in days).

  Raises
  ------
  Raises a ``IndexError`` if the index is out of bounds.
  )doc";

static const auto DOC_ImageStack_build_zeroed_times = R"doc(
  Construct an array of time differentials between each image
  in the stack and the first image. This can return negative times
  if the images are not sorted by time.

  zeroed_time[i] = time[i] - time[0] 

  Returns
  -------
  zeroed_times : `list`
      A list of times starting at 0.0.
  )doc";

static const auto DOC_ImageStack_sort_by_time = R"doc(
  Sort the images in the ImageStack by their time.

  Raises
  ------
  Raises a ``RuntimeError`` if the input image is the data is currently on GPU.    
  )doc";

static const auto DOC_ImageStack_convolve_psf = R"doc(
  Convolves each image (science and variance layers) with the PSF
  stored in the LayeredImage object.
  )doc";

static const auto DOC_ImageStack_get_width = R"doc(
  Returns the width of the images in pixels.
   
  Returns
  -------
  npixels : `int`
      The width of each image in pixels.
  )doc";

static const auto DOC_ImageStack_get_height = R"doc(
  Returns the height of the images in pixels.
   
  Returns
  -------
  npixels : `int`
      The height of each image in pixels.
  )doc";

static const auto DOC_ImageStack_get_npixels = R"doc(
  Returns the number of pixels per image.
   
  Returns
  -------
  npixels : `int`
      The number of pixels per image.
  )doc";

static const auto DOC_ImageStack_get_total_pixels = R"doc(
  Returns the total number of pixels in all the images.
   
  Returns
  -------
  npixels : `int`
      The total number of pixels over all images.
  )doc";

static const auto DOC_ImageStack_copy_to_gpu = R"doc(
  Make a copy of the image and time data on the GPU. The image data
  is stored as a single linear vector of floats where the value of
  pixel (``i``, ``j``) in the image at time ``t`` is at:
  ``index = t * width * height + i * width + j``
  )doc";

static const auto DOC_ImageStack_clear_from_gpu = R"doc(
  Frees both the time and image data from the GPU.
  )doc";

}  // namespace pydocs

#endif /* IMAGESTACK_DOCS */
