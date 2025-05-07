#ifndef IMAGE_UTILS_CPP_DOCS_H_
#define IMAGE_UTILS_CPP_DOCS_H_

namespace pydocs {

static const auto DOC_image_utils_cpp_convolve_cpu = R"doc(
  Convolve the image with a PSF on the CPU.

  Parameters
  ----------
  image : `numpy.ndarray`
      The image data as a two dimensional array.
  psf : `numpy.ndarray`
      The kernel of the Point Spread Function as a two dimensional array.

  Returns
  -------
  result : `numpy.ndarray`
      The resulting image.
  )doc";

static const auto DOC_image_utils_cpp_convolve_gpu = R"doc(
  Convolve the image with a PSF on the GPU.

  Parameters
  ----------
  image : `numpy.ndarray`
      The image data as a two dimensional array.
  psf : `numpy.ndarray`
      The kernel of the Point Spread Function as a two dimensional array.

  Returns
  -------
  result : `numpy.ndarray`
      The resulting image.
  )doc";

static const auto DOC_image_utils_cpp_convolve = R"doc(
  Convolves the image (in place) with a PSF using a CPU if one is
  available and a GPU otherwise.

  Parameters
  ----------
  image : `numpy.ndarray`
      The image data as a two dimensional array.
  psf : `numpy.ndarray`
      The kernel of the Point Spread Function as a two dimensional array.

  Returns
  -------
  result : `numpy.ndarray`
      The resulting image.
)doc";

static const auto DOC_image_utils_square_psf = R"doc(
  Compute the (unnormalized) square of a psf.

  Parameters
  ----------
  given_psf : `numpy.ndarray`
      The kernel of the Point Spread Function as a two dimensional array.

  Returns
  -------
  result : `numpy.ndarray`
      The resulting kernel.
)doc";

static const auto DOC_image_utils_generate_psi_image = R"doc(
  Generates the full psi image where the value of each pixel p in the
  resulting image is science[p] / variance[p], skipping masked pixels.
  Convolves the resulting image with the PSF.

  Parameters
  ----------
  sci : `numpy.ndarray`
      The science data as a H x W dimensional array.
  var : `numpy.ndarray`
      The variance data as a H x W dimensional array.
  psf : `numpy.ndarray`
      The kernel of the Point Spread Function as a two dimensional array.
      
  Returns
  -------
  result : `numpy.ndarray`
      A numpy array the same shape as the input image.
  )doc";

static const auto DOC_image_utils_generate_phi_image = R"doc(
  Generates the full psi image where the value of each pixel p in the
  resulting image is 1.0 / variance[p], skipping masked pixels.

  Convolves the resulting image with the square of the PSF.

  Parameters
  ----------
  var : `numpy.ndarray`
      The variance data as a H x W dimensional array.
  psf : `numpy.ndarray`
      The kernel of the Point Spread Function as a two dimensional array.
      
  Returns
  -------
  result : `numpy.ndarray`
      A numpy array the same shape as the input image.
  )doc";

} /* namespace pydocs */

#endif /* IMAGE_UTILS_CPP_DOCS_H_ */
