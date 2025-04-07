#ifndef LAYEREDIMAGE_DOCS
#define LAYEREDIMAGE_DOCS

namespace pydocs {
static const auto DOC_LayeredImage = R"doc(
  Creates a layered_image out of individual `RawImage` layers.

  Parameters
  ----------
  sci : `RawImage`
      The `RawImage` for the science layer.
  var : `RawImage`
      The `RawImage` for the cariance layer.
  msk : `RawImage`
      The `RawImage` for the mask layer.
  psf : `numpy.ndarray`
      The kernel of the PSF.
  obstime : `float`
      The time of the image (in UTC MJD).

  Raises
  ------
  RuntimeError:
      If the science, variance and mask are not the same size.
  )doc";

static const auto DOC_LayeredImage_set_psf = R"doc(
  Sets the PSF kernel.
  )doc";

static const auto DOC_LayeredImage_get_psf = R"doc(
  Returns the PSF kernel.
  )doc";

static const auto DOC_LayeredImage_mask_pixel = R"doc(
  Apply masking to a single pixel. Applies to all three layers so that
  it can be used before or after ``apply_mask()``. 

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Col index.
  )doc";

static const auto DOC_LayeredImage_binarize_mask = R"doc(
  Convert the bitmask of flags into a single binary value of 1
  for pixels that match one of the flags to use and 0 otherwise.
  Modifies the mask layer in-place. Used to select which masking
  flags are applied.

  Note: This is a no-op for masks that are already binary and it is
  safe to call this function multiple times.

  Parameters
  ----------
  flags_to_use : `int`
      The bit mask of mask flags to keep.
  )doc";

static const auto DOC_LayeredImage_apply_mask = R"doc(
  Applies the mask layer to each of the science and variance layers
  by checking whether the pixel in the mask layer is 0 (no masking)
  or non-zero (masked). Applies all flags. To use a subset of flags
  call binarize_mask() first.
  )doc";

static const auto DOC_LayeredImage_get_science = R"doc(
  Returns the science layer `RawImage`.
  )doc";

static const auto DOC_LayeredImage_get_mask = R"doc(
  Returns the mask layer `RawImage`.
  )doc";

static const auto DOC_LayeredImage_get_variance = R"doc(
  Returns the variance layer `RawImage`.
  )doc";

static const auto DOC_LayeredImage_set_science = R"doc(
  Returns the science layer `RawImage`.
  )doc";

static const auto DOC_LayeredImage_set_mask = R"doc(
  Returns the mask layer `RawImage`.
  )doc";

static const auto DOC_LayeredImage_set_variance = R"doc(
  Returns the science layer `RawImage`.
  )doc";

static const auto DOC_LayeredImage_convolve_psf = R"doc(
  Convolves the PSF stored within the LayeredImage with the science and variance
  layers (uses the PSF-squared for the variance). Modifies the layers in place.
  )doc";

static const auto DOC_LayeredImage_convolve_given_psf = R"doc(
  Convolves a given PSF with the science and variance layers
  (uses the PSF-squared for the variance). Modifies the layers in place.

  Parameters
  ----------
  psf : `PSF`
      The PSF to use.
  )doc";

static const auto DOC_LayeredImage_get_width = R"doc(
  Returns the image's width in pixels.
  )doc";

static const auto DOC_LayeredImage_get_height = R"doc(
  Returns the image's height in pixels.
  )doc";

static const auto DOC_LayeredImage_get_npixels = R"doc(
  Returns the image's total number of pixels.
  )doc";

static const auto DOC_LayeredImage_get_obstime = R"doc(
  Get the image's observation time in UTC MJD.
  )doc";

static const auto DOC_LayeredImage_set_obstime = R"doc(
  Set the image's observation time in UTC MJD.
  )doc";

static const auto DOC_LayeredImage_contains = R"doc(
  Returns a Boolean indicating whether the image contains the given coordinates.

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Col index.

  Returns
  -------
  result : `bool`
      A Boolean indicating whether the image contains the given coordinates.
  )doc";

static const auto DOC_LayeredImage_get_science_pixel = R"doc(
  Get the science pixel value at given index, checking the mask layer.
  Returns NO_DATA if any of the mask bits are set.

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Col index.

  Returns
  -------
  value : `float`
      Pixel value.
  )doc";

static const auto DOC_LayeredImage_get_variance_pixel = R"doc(
  Get the variance pixel value at given index, checking the mask layer.
  Returns NO_DATA if any of the mask bits are set.

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Col index.

  Returns
  -------
  value : `float`
      Pixel value.
  )doc";

static const auto DOC_LayeredImage_science_pixel_has_data = R"doc(
  Checks whether the science pixel has valid data by checking both
  the value in the science and mask layers.

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Col index.

  Returns
  -------
  value : `bool`
      Whether the science pixel has data.
  )doc";

static const auto DOC_LayeredImage_generate_psi_image = R"doc(
  Generates the full psi image where the value of each pixel p in the
  resulting image is science[p] / variance[p]. To handle masked bits
  apply_mask() must be called before the psi image is generated. Otherwise,
  all pixels are used.
  Convolves the resulting image with the PSF.

  Returns
  -------
  result : `kbmod.RawImage`
      A ``RawImage`` of the same dimensions as the ``LayeredImage``.
  )doc";

static const auto DOC_LayeredImage_generate_phi_image = R"doc(
  Generates the full phi image where the value of each pixel p in the
  resulting image is 1.0 / variance[p]. To handle masked bits
  apply_mask() must be called before the phi image is generated. Otherwise,
  all pixels are used.
  Convolves the resulting image with the PSF.

  Returns
  -------
  result : `kbmod.RawImage`
      A ``RawImage`` of the same dimensions as the ``LayeredImage``.
  )doc";

}  // namespace pydocs

#endif /* LAYEREDIMAGE_DOCS  */
