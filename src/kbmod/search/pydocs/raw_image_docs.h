#ifndef RAWIMAGEDOCS
#define RAWIMAGEDOCS

namespace pydocs {
static const auto DOC_RawImage = R"doc(
  Image and the time it was observed at.

  Instantiated from an image or from image dimensions and and a value (which
  defaults to zero).

  Parameters
  ----------
  image : `numpy.array`, optional
      Image, row-major a 2D array. The array *must* be of dtype `numpy.single`.
  width : `int`, optional
      Width of the image in pixels.
  height : `int`, optional
      Height of the image in pixels.
  value : `float`, optional
      When instantiated from dimensions, value that fills the array.
      Default is 0.

  Attributes
  ----------
  height : `int`
      Image height, in pixels.
  width : `int`
      Image width, in pixels.
  npixels : `int`
      Number of pixels in the image, equivalent to ``width*height``.
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
  methods, such as `add_fake_object`, however, use the `(x, y)` convention which
  is the reversed NumPy convention. Refer to individual methods signature and docstring
  to see which one they use.

  Examples
  --------
  >>> from kbmod.search import RawImage
  >>> import numpy as np
  >>> ri = RawImage()
  >>> ri = RawImage(w=2, h=3, value=1)
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
      Row index (y position)
  j : `int`
      Col index (x position)

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
      Row index (y position)
  j : `int`
      Col index (x position)

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
      Row index (y position)
  j : `int`
      Col index (x position)
  value : `float`
      Value to set the pixels to.
  )doc";

static const auto DOC_RawImage_contains_index = R"doc(
  True if the given index falls within the image dimensions.
  Note that the x and y ordering is the inverse of contains_point().

  Parameters
  ----------
  i : `int`
      Row index (y position)
  j : `int`
      Col index (x position)

  Returns
  -------
  result : `bool`
      ``True`` when point falls within the image dimensions.
  )doc";

static const auto DOC_RawImage_contains_point = R"doc(
  True if the given point falls within the image dimensions.
  Note that the x and y ordering is the inverse of contains_index().
      
  Parameters
  ----------
  x : `float`
      The real valued x position (mapped to the matrix's column).
  y : `float`
      The real valued y position (mapped to the matrix's row).

  Returns
  -------
  result : `bool`
      ``True`` when point falls within the image dimensions.
  )doc";

static const auto DOC_RawImage_mask_pixel = R"doc(
  Sets image pixel at an invalid value that indicates it is masked.

  Parameters
  ----------
  i : `int`
      Row index (y position)
  j : `int`
      Col index (x position)
  )doc";

static const auto DOC_RawImage_set_all = R"doc(
  Sets all image pixel values to the given value.

  Parameters
  ----------
  value : `float`
      Value to set the pixels to.
  )doc";

static const auto DOC_RawImage_replace_masked_values = R"doc(
  Replace the masked values in an image with a given value.

  Parameters
  ----------
  value : `float`
      The value to swap in. Default = 0.0.
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

  Parameters
  ----------
  strict_checks : `bool`
      If True and none of the pixels contain data, then raises an RuntimeError.
      If False and none of the pixels contain data, returns (0.0, 0.0).

  Returns
  -------
  bounds : `tuple`
      A ``(min, max)`` tuple.
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

} /* namespace pydocs */

#endif /* RAWIMAGE_DOCS */
