#ifndef RAWIMAGEDOCS
#define RAWIMAGEDOCS

namespace pydocs{
  static const auto DOC_RawImage = R"doc(
  Raw Image, a row-ordered 2D array unraveled into a vector.
  )doc";

  static const auto DOC_RawImage_get_height = R"doc(
  Returns the image's height in pixels.
  )doc";

  static const auto DOC_RawImage_get_width = R"doc(
  Returns the image's width in pixels.
  )doc";

  static const auto DOC_RawImage_get_npixels = R"doc(
  Returns the image's total number of pixels.
  )doc";

  static const auto DOC_RawImage_get_all_pixels = R"doc(
  Returns a list of the images pixels.
  )doc";

  static const auto DOC_RawImage_set_array = R"doc(
  Sets all image pixels given an array of values.
  )doc";

  static const auto DOC_RawImage_get_obstime = R"doc(
  Get the observation time of the image.
  )doc";

  static const auto DOC_RawImage_set_obstime = R"doc(
  Set the observation time of the image.
  )doc";

  static const auto DOC_RawImage_approx_equal = R"doc(
  Checks if two images are approximately equal.
  )doc";

  static const auto DOC_RawImage_compute_bounds = R"doc(
  Returns min and max pixel values.
  )doc";

  static const auto DOC_RawImage_find_peak = R"doc(
  Returns the pixel coordinates of the maximum value.
  )doc";

  static const auto DOC_RawImage_find_central_moments = R"doc(
  Returns the central moments of the image.
  )doc";

  static const auto DOC_RawImage_create_stamp = R"doc(
  Create an image stamp around a given region.

  Parameters
  ----------
  x : `float`
      The x value of the center of the stamp.
  y : `float`
      The y value of the center of the stamp.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  interpolate : `bool`
      A Boolean indicating whether to interpolate pixel values.
  keep_no_data : `bool`
      A Boolean indicating whether to preserve NO_DATA tags or to
      replace them with 0.0.
  
  Returns
  -------
  `RawImage`
      The stamp.
  )doc";

  static const auto DOC_RawImage_set_pixel = R"doc(
  Set the value of a given pixel.
  )doc";

  static const auto DOC_RawImage_add_pixel = R"doc(
  Add to the raw value of a given pixel.
  )doc";

  static const auto DOC_RawImage_add_pixel_interp = R"doc(
  Add to the value calculated by bilinear interpolation
  of the neighborhood of the given pixel position.
  )doc";

  static const auto DOC_RawImage_apply_mask = R"doc(
  Applies a mask to the RawImage by comparing the given bit vector with the
  values in the mask layer and marking pixels NO_DATA. Modifies the image in-place.

  Parameters
  ----------
  flag : `int`
      The bit mask of mask flags to use.
  exceptions : `list` of `int`
      A list of exceptions (combinations of bits where we do not apply the mask).
  mask : `RawImage`
      The image of pixel mask values.
  )doc";

  static const auto DOC_RawImage_grow_mask = R"doc(
  Expands the NO_DATA tags to nearby pixels. Modifies the image in-place.

  Parameters
  ----------
  steps : `int`
     The number of pixels by which to grow the masked regions.
  )doc";

  static const auto DOC_RawImage_pixel_has_data = R"doc(
  Returns a Boolean indicating whether the pixel has data.
  )doc";

  static const auto DOC_RawImage_set_all = R"doc(
  Set all pixel values given an array.
  )doc";

  static const auto DOC_RawImage_get_pixel = R"doc(
  Returns the value of a pixel.
  )doc";

  static const auto DOC_RawImage_get_pixel_interp =  R"doc(
  Get the interoplated value of a pixel.
  )doc";

  static const auto DOC_RawImage_convolve = R"doc(
  Convolve the image with a PSF.
  )doc";

  static const auto DOC_RawImage_convolve_cpu = R"doc(
  Convolve the image with a PSF.
  )doc";

  static const auto DOC_RawImage_load_fits = R"doc(
  Load the image data from a FITS file.
  )doc";

  static const auto DOC_RawImage_save_fits = R"doc(
  Save the image to a FITS file.
  )doc";

  static const auto DOC_RawImage_append_fits_layer = R"doc(
  Append the image as a layer in a FITS file.
  )doc";

} /* namespace pydocs */

#endif /* RAWIMAGE_DOCS */
