#ifndef RAWIMAGEDOCS
#define RAWIMAGEDOCS

namespace pydocs {
static const auto DOC_RawImage = R"doc(
  Image and the time it was observed at.

  Instantiated from a pair of obstime and image, or from image dimensions and
  obstime. When instantiating from image dimensions and obstime, it's possible
  to provide a default value used to fill the array with. Otherwise the array is
  filled with zeros.

  Parameters
  ----------
  image : `numpy.array`, optional
      Image, row-major a 2D array. The array *must* be of dtype `numpy.single`.
  obstime : `float`, optional
      MJD stamp, time the image was observed at.
  width : `int`, optional
      Width of the image.
  height : `int`, optional
      Height of the image.
  value : `float`, optional
      When instantiated from dimensions and obstime, value that fills the array.
      Default is 0.

  Attributes
  ----------
  height : `int`
      Image height, in pixels.
  width : `int`
      Image width, in pixels.
  npixels : `int`
      Number of pixels in the image, equivalent to ``width*height``.
  obstime : `float`
      MJD time of observation.
  image : `np.array[np,single]`
      Image array.

  Notes
  -----
  RawImage is internally represented by an Eigen_ Matrix object that uses
  ``float`` type. Because of this the given numpy array **must** be of
  `np.single` dtype. This is on purpose since memory on a GPU comes at a
  premium. Tests determined that loss of precision does not greatly affect the
  search.

  Note also that KBMOD uses ``(width, height)`` convention is opposite to the
  NumPy' `array.shape` convention which uses `(row, col)`. KBMOD also
  distinguishes between a pair of coordinates in Cartesian plane, i.e. a point,
  which, usually expressed with the ``(x, y)`` convention and a pair of values
  representing indices to a 2D matrix, usually expressed with the ``(i, j)``
  convention. Pixel accessing or setting methods of `RawImage`, such as
  `get_pixel`, use the indexing convention. This matches NumPy convention. Other
  methods, such as `interpolate` or `add_fake_object`, however, use the `(x, y)`
  convention which is the reversed NumPy convention. Refer to individual methods
  signature and docstring to see which one they use.

  Examples
  --------
  >>> from kbmod.search import RawImage
  >>> import numpy as np
  >>> ri = RawImage()
  >>> ri = RawImage(w=2, h=3, value=1, obstime=10)
  >>> ri.image
  array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)
  >>> ri = RawImage(np.zeros((2, 3), dtype=np.single), 10)
  >>> ri.image
  array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32)

  .. _Eigen: https://eigen.tuxfamily.org/index.php?title=Main_Page
  )doc";

static const auto DOC_RawImage_get_pixel = R"doc(
  Get pixel at given index.

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

static const auto DOC_RawImage_pixel_has_data = R"doc(
  True if the pixel at given index is not masked.

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Col index.

  Returns
  -------
  has_data : `bool`
      `True` when pixel is not masked, `False` otherwise.
  )doc";

static const auto DOC_RawImage_set_pixel = R"doc(
  Sets image pixel at given index to the given value.

  Parameters
  ----------
  i : `int`
      Row.
  j : `int`
      Column.
  value : `float`
      Value to set the pixels to.
  )doc";

static const auto DOC_RawImage_set_all = R"doc(
  Sets all image pixel values to the given value.

  Parameters
  ----------
  value : `float`
      Value to set the pixels to.
  )doc";

static const auto DOC_RawImage_l2_allclose = R"doc(
  `True` when L2 norm of the two arrays is within the given precision.

  Parameters
  ----------
  other : `RawImage`
      Image to compare this image to.

  Returns
  -------
  approx_equal : `bool`
      `True` if ``||this - other|| < atol``, `False` otherwise.
  )doc";

static const auto DOC_RawImage_compute_bounds = R"doc(
  Returns min and max pixel values, ignoring the masked pixels.

  Returns
  -------
  bounds : `tuple`
      A ``(min, max)`` tuple.
  )doc";

static const auto DOC_RawImage_find_peak = R"doc(
  Returns the pixel coordinates of the maximum value.

  Parameters
  ----------
  furthest_from_center : `bool`
      When `True`, and multiple identical maxima exist, returns the one that is
      at a greatest L2 distance from the center of the image. Otherwise it
      returns the last maxima that was found in a row-wise ordered search.

  Returns
  -------
  location : `Index`, optional
      Index of the maximum.
  )doc";

static const auto DOC_RawImage_find_central_moments = R"doc(
  Returns the central moments of the image.

  Returns
  -------
  moments : `ImageMoments`
      Image moments.
  )doc";

static const auto DOC_RawImage_center_is_local_max = R"doc(
  A filter on whether the center of the stamp is a local
    maxima and the percentage of the stamp's total flux in this
    pixel.

  Parameters
  ----------
  local_max : ``bool``
    Require the central pixel to be a local maximum.
  flux_thresh : ``float``
    The fraction of the stamp's total flux that needs to be in
    the center pixel [0.0, 1.0].

  Returns
  -------
  keep_row : `bool`
      Whether or not the stamp passes the check.
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
  keep_no_data : `bool`
      A Boolean indicating whether to preserve NO_DATA tags or to
      replace them with 0.0.

  Returns
  -------
  `RawImage`
      The stamp.
  )doc";

static const auto DOC_RawImage_interpolate = R"doc(
  Get the interoplated value of a point.

  Parameters
  ----------
  x : `float`
      The x-coordinate, the abscissa.
  y : `float`
      The y-coordinate, the ordinate.

  Returns
  -------
  value : `float`
      Bilinearly interpolated value at that point.

  )doc";

static const auto DOC_RawImage_interpolated_add = R"doc(
  Add the given value at a given point, to the image.

  Addition is performed by determining the nearest Manhattan neighbors, weighing
  the given value by the distance to these neighbors and then adding that value
  at the index locations of the neighbors. Sort of like an inverse bilinear
  interpolation.

  Parameters
  ----------
  x : `float`
      The x coordinate at which to add value.
  y : `float`
      Y coordinate.
  value : `float`
      Value to add.
  )doc";

static const auto DOC_RawImage_get_interp_neighbors_and_weights = R"doc(
  Returns a tuple of Manhattan neighbors and interpolation weights.

  Parameters
  ----------
  x : `float`
      The x coordinate at which to add value.
  y : `float`
      Y coordinate.
  )doc";

static const auto DOC_RawImage_apply_mask = R"doc(
  Applies a mask to the RawImage by comparing the given bit vector with the
  values in the mask layer and marking pixels ``NO_DATA``.

  Modifies the image in-place.

  Parameters
  ----------
  flag : `int`
      The bit mask of mask flags to use. Use 0xFFFFFF to apply all flags.
  mask : `RawImage`
      The image of pixel mask values.
  )doc";

static const auto DOC_RawImage_convolve_gpu = R"doc(
  Convolve the image with a PSF on the GPU.

  Convolves in-place.

  Parameters
  ----------
  psf : `PSF`
      Point Spread Function.
  )doc";

static const auto DOC_RawImage_convolve_cpu = R"doc(
  Convolve the image with a PSF.

  Convolves in-place.

  Parameters
  ----------
  psf : `PSF`
      Point Spread Function.
  )doc";

static const auto DOC_RawImage_load_fits = R"doc(
  Load the image data from a FITS file.

  Parameters
  ----------
  path : `str`
      Path to an existing FITS file.
  ext : `int`, optional
      Extension index. Default: 0.
  )doc";

static const auto DOC_RawImage_save_fits = R"doc(
  Save the image to a FITS file.

  Parameters
  ----------
  path : `str`
      Path to the new file.
  )doc";

static const auto DOC_RawImage_append_to_fits = R"doc(
  Append the image as a layer in a FITS file.

  Parameters
  ----------
  path : `str`
      Path to an existing file.
  )doc";

} /* namespace pydocs */

#endif /* RAWIMAGE_DOCS */
