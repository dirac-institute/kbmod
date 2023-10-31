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

static const auto DOC_LayeredImage_apply_mask_flags = R"doc(
  Applies a mask to each image by comparing the given bit vector with the
  values in the mask layer and marking pixels NO_DATA. 
  Modifies the science and variance layers in-place.

  Parameters
  ----------
  flag : `int`
      The bit mask of mask flags to use.
  exceptions : `list` of `int`
      A list of exceptions (combinations of bits where we do not apply the mask).
  )doc";

static const auto DOC_LayeredImage_apply_mask_threshold = R"doc(
  Applies a threshold mask by setting pixel values over a given threshold
  to NO_DATA. Modifies the science and variance layers in-place.

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

static const auto DOC_LayeredImage_generate_psi_image = R"doc(
  todo
  )doc";

static const auto DOC_LayeredImage_generate_phi_image = R"doc(
  todo
  )doc";
}  // namespace pydocs

#endif /* LAYEREDIMAGE_DOCS  */
