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

} /* namespace pydocs */

#endif /* IMAGE_UTILS_CPP_DOCS_H_ */
