#ifndef GEOM_DOCS
#define GEOM_DOCS

namespace pydocs {

static const auto DOC_Index = R"doc(
  Array index.

  Index can be compared to tuples and cast to a NumPy structured array.
  Index will cast non-int types into an integer without rounding, i.e.
  without applying ``floor`` or ``ceil`` to round the value to the
  nearest integer.

  Parameters
  ----------
  i : `int`
      Row index.
  j : `int`
      Column index.
  )doc";

static const auto DOC_Index_to_yaml = R"doc(
  Returns a single YAML record.
  )doc";

static const auto DOC_Point = R"doc(
  A point in Cartesian plane.

  Point can be compared to tuples and cast to a NumPy structured array.

  Parameters
  ----------
  x : `float`
      Row index.
  y : `float`
      Column index.
  )doc";

static const auto DOC_Point_to_index = R"doc(
  Returns the `Index` this point is located in.
  )doc";

static const auto DOC_Point_to_yaml = R"doc(
  Returns a single YAML record.
  )doc";

static const auto DOC_Rectangle = R"doc(
  A rectangular selection of an array.

  The rectangle can also contain its corner origin `Index` with respect to a
  second reference point. Most commonly the corner of another, larger, rectangle
  - f.e. as is the case when selecting a stamp to copy from origin array and
  pasting the selection into a destination array.

  Rectangles can be cast into NumPy structured arrays.

  Parameters
  ----------
  corner : `Index` or `tuple`
      Top left corner of the rectangle, in origin coordinates.
  anchor : `Index` or `tuple`, optional
      Top left corner of the rectangle, in destination coordinates.
  width : `int`
      Positive integer, width of the rectangle.
  height : `int`
      Positive integer, height of the rectangle.

  Attributes
  ----------
  i : `int`
      Row index of the corner.
  j : `int`
      Column index of the corner.
  )doc";

static const auto DOC_Rectangle_to_yaml = R"doc(
  Returns a single YAML record.
  )doc";

static const auto DOC_centered_range = R"doc(
  Given a reference value and a radius around it, returns the range start, end
  and length that fit into the width of the interval.

  The returned range is [val-r, val+r].

  Parameters
  ----------
  val : `int`
      Center of the returned range.
  r : `int`
      Radius around the center to select.
  width : `int`
      Maximum allowed width of the interval.

  Returns
  -------
  interval : `tuple`
      The triplet (start, end, length) = [val-r, val+r, 2r+1] trimmed to fit
      within [0, width] range.

  Examples
  --------
  Interval of radius 1, centered on 5, i.e. [4, 5, 6]:

  >>> centered_range(5, 1, 10)
  (4, 6, 3)

  Interval of radius 2, centered on 1, this time the allowed range [0, width]
  clips the returned range, i.e. [0, 1, 2, 3]

  >>> centered_range(1, 2, 10)
  (0, 3, 4)
  )doc";

static const auto DOC_anchored_block = R"doc(
  Returns rectangle selection of an array centered on given coordinate.

  Parameters
  ----------
  idx : `tuple`
      Center of the rectangle selection, indices ``(row, col)``.
  r : `int`
      Radius around the central index to select.
  shape : `int`
      Shape of the origin array.

  Returns
  -------
  rect : `Rectangle`
      Selected rectangle, such that the corner + width/height returns the
      desired array slice.

  Examples
  --------
  >>> img = numpy.arange(100).reshape(10, 10)
  >>> rect = anchored_block((5, 5), 1, img.shape)
  >>> rect
  Rectangle(corner: (4, 4), anchor: (0, 0), width: 3, height: 3)
  >>> stamp = img[rect.i:rect.i+rect.height, rect.j:rect.j+width]
  >>> stamp
  array([[44, 45, 46],
         [54, 55, 56],
         [64, 65, 66]])

  By default, anchor is calculated such that it fits into a destination with a
  shape ``(2r+1, 2r+1)``, i.e. ``(rect.width, rect.height)``. Note the
  requested radius clips to the left and top of the array.

  >>> dest = np.zeros((3, 3)) - 1
  >>> rect = anchored_block((0, 0), 1, img.shape)
  >>> dest[rect.anchor.i:, rect.anchor] = stamp
  array([[-1., -1., -1.],
         [-1.,  0.,  1.],
         [-1., 10., 11.]])
  )doc";

}  // namespace pydocs

#endif  // GEOM_DOCS
