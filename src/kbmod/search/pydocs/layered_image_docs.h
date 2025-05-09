#ifndef LAYEREDIMAGE_DOCS
#define LAYEREDIMAGE_DOCS

namespace pydocs {
static const auto DOC_LayeredImage = R"doc(
  Creates a layered_image out of individual image layers. The LayeredImage
  takes ownership of each layer, invalidating the previous object.

  Attributes
  ----------
  height : `int`
      Image height, in pixels. Read only in Python.
  width : `int`
      Image width, in pixels. Read only in Python.
  time : `float`
      The time of the image (in UTC MJD).
  sci : `numpy.ndarray`
      The array for the science layer. Getter only in Python.
  var : `numpy.ndarray`
      The array for the variance layer. Getter only in Python.
  mask : `numpy.ndarray`
      The array for the mask layer. Getter only in Python.
  psf : `numpy.ndarray`
      The kernel of the PSF.
      
  Parameters
  ----------
  sci : `numpy.ndarray`
      The data for the science layer. The LayeredImage takes ownership.
  var : `numpy.ndarray`
      The data for the variance layer. The LayeredImage takes ownership.
  msk : `numpy.ndarray`
      The data for the mask layer. The LayeredImage takes ownership.
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

static const auto DOC_LayeredImage_apply_mask = R"doc(
  Applies the mask layer to each of the science and variance layers
  by checking whether the pixel in the mask layer is 0 (no masking)
  or non-zero (masked). Applies all flags. To use a subset of flags
  call binarize_mask() first.
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

static const auto DOC_LayeredImage_generate_psi_image = R"doc(
  Generates the full psi image where the value of each pixel p in the
  resulting image is science[p] / variance[p]. To handle masked bits
  apply_mask() must be called before the psi image is generated. Otherwise,
  all pixels are used.

  Convolves the resulting image with the PSF.

  Returns
  -------
  result : `numpy.ndarray`
      A numpy array the same shape as the input image.
  )doc";

static const auto DOC_LayeredImage_generate_phi_image = R"doc(
  Generates the full phi image where the value of each pixel p in the
  resulting image is 1.0 / variance[p]. To handle masked bits
  apply_mask() must be called before the phi image is generated. Otherwise,
  all pixels are used.

  Convolves the resulting image with the square of the PSF.

  Returns
  -------
  result : `numpy.ndarray`
      A numpy array the same shape as the input image.
  )doc";

}  // namespace pydocs

#endif /* LAYEREDIMAGE_DOCS  */
