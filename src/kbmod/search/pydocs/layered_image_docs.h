#ifndef LAYEREDIMAGE_DOCS
#define LAYEREDIMAGE_DOCS

namespace pydocs {
static const auto DOC_LayeredImage = R"doc(
  Creates a layered_image out of individual `RawImage` layers.

  Parameters
  ----------
  path : `str`, optional
      Path to a FITS file containing ``science``, ``mask`` and ``variance``
      extensions.
  sci : `RawImage`, optional
      The `RawImage` for the science layer.
  var : `RawImage`, optional
      The `RawImage` for the cariance layer.
  msk : `RawImage`, optional
      The `RawImage` for the mask layer.
  name : `str`, optional
      File/layered image name.
  width : `int`, optional
      Width of the images (in pixels).
  height : `int`, optional
      Height of the images (in pixels).
  std: `float`, optional
      Standard deviation of the image.
  var: `float`, optional
      Variance of the pixels, assumed uniform.
  time : `float`, optional
      Observation time.
  seed : `int`, optional
      Pseudorandom number generator.

  psf : `PSF`
      The PSF for the image.

  Raises
  ------
  RuntimeError:
      If the science, variance and mask are not the same size.

  Notes
  -----
  Class can be instantiated from a file, or from 3 `RawImage` (science, mask,
  variance) objects, or by providing the name, dimensions, standard deviation,
  variance, observation time to which, additionally, a random seed generator can
  be provided. PSF is always required.
  )doc";

static const auto DOC_LayeredImage_set_psf = R"doc(
  Sets the PSF object.
  )doc";

static const auto DOC_LayeredImage_get_psf = R"doc(
  Returns the PSF object.
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

static const auto DOC_LayeredImage_union_masks = R"doc(
  Unions the masked pixel flags from the a given second mask layer onto
  this image's mask layer. Modifies the mask layer in place.

  Parameters
  ----------
  global_mask : `RawImage`
      The `RawImage` of global mask values (binary) for each pixel.
  )doc";

static const auto DOC_LayeredImage_union_threshold_masking = R"doc(
  Masks any pixel whose corresponding value in the science layer is
  above the given threshold using mask flag = 1.

  Parameters
  ----------
  thresh : `float`
      The threshold value to use.
  )doc";

static const auto DOC_LayeredImage_sub_template = R"doc(
  Subtract given image template
  )doc";

static const auto DOC_LayeredImage_save_layers = R"doc(
  Saves the LayeredImage to a FITS file with layers for the science,
  mask, and variance.

  Saves the file as {path}/{filename}.fits where the path is given
  and the file name is an object attribute.

  Parameters
  ----------
  path : `str`
      The file path to use. 
  )doc";

static const auto DOC_LayeredImage_get_science = R"doc(
  Returns the science layer raw_image.
  )doc";

static const auto DOC_LayeredImage_get_mask = R"doc(
  Returns the mask layer raw_image.
  )doc";

static const auto DOC_LayeredImage_get_variance = R"doc(
  Returns the variance layer raw_image.
  )doc";

static const auto DOC_LayeredImage_set_science = R"doc(
  Returns the science layer raw_image.
  )doc";

static const auto DOC_LayeredImage_set_mask = R"doc(
  Returns the mask layer raw_image.
  )doc";

static const auto DOC_LayeredImage_set_variance = R"doc(
  Returns the science layer raw_image.
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

static const auto DOC_LayeredImage_grow_mask = R"doc(
  Expands the NO_DATA tags to nearby pixels in the science and variance layers.
  Modifies the images in-place.

  Parameters
  ----------
  steps : `int`
     The number of pixels by which to grow the masked regions.
  )doc";

static const auto DOC_LayeredImage_get_name = R"doc(
  Returns the name of the layered image.
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
  Get the image's observation time.
  )doc";

static const auto DOC_LayeredImage_set_obstime = R"doc(
  Set the image's observation time.
  )doc";

static const auto DOC_LayeredImage_cointains = R"doc(
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

static const auto DOC_LayeredImage_generate_psi_image = R"doc(
  todo
  )doc";

static const auto DOC_LayeredImage_generate_phi_image = R"doc(
  todo
  )doc";
}  // namespace pydocs

#endif /* LAYEREDIMAGE_DOCS  */
