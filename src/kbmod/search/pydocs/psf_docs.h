#ifndef PSF_DOCS
#define PSF_DOCS

namespace pydocs {
static const auto DOC_PSF = R"doc(
  Point Spread Function.

  Parameters
  ----------
  stdev : `float`, optional
      Standard deviation of the Gaussian PSF. Must be > 0.0.
  psf : `PSF`, optional
      Another PSF object.
  arr : `numpy.array`, optional
      A realization of the PSF.

  Notes
  -----
  When instantiated with another `psf` object, returns its copy.
  When instantiated with an array-like object, that array must be
  a square matrix and have an odd number of dimensions. Only one
  of the arguments is required.

  Raises
  ------
  Raises a ``RuntimeError`` when given an invalid stdev or an array
  containing invalid entries, such as NaN or infinity.
  )doc";

static const auto DOC_PSF_set_array = R"doc(
  Set the kernel values of a realized PSF.

  Parameters
  ----------
  arr : `numpy.array`
      A realization of the PSF.

  Notes
  -----
  Given realization of a PSF has to be an odd-dimensional square
  matrix.
  )doc";

static const auto DOC_PSF_get_sum = R"doc(
  "Returns the sum of PSFs kernel elements.
  ")doc";

static const auto DOC_PSF_get_dim = R"doc(
  "Returns the PSF kernel dimension D where the kernel is a D by D array.
  ")doc";

static const auto DOC_PSF_get_radius = R"doc(
  "Returns the radius of the PSF.
  ")doc";

static const auto DOC_PSF_get_size = R"doc(
  "Returns the number of elements in the PSFs kernel.
  ")doc";

static const auto DOC_PSF_get_kernel = R"doc(
  "Returns the PSF kernel.
  ")doc";

static const auto DOC_PSF_get_value = R"doc(
  "Returns the PSF kernel value at a specific point.
  ")doc";

static const auto DOC_PSF_square_psf = R"doc(
  "Squares, raises to the power of two, the elements of the PSF kernel.
  ")doc";

static const auto DOC_PSF_print = R"doc(
  "Pretty-prints the PSF.
  ")doc";
}  // namespace pydocs

#endif /* PSF_DOCS */
