#ifndef GEOM_H_
#define GEOM_H_

#include <iostream>
#include <optional>
#include <utility>  // pair
#include <array>
#include <vector>
#include <assert.h>

#include <Eigen/Core>

#include "common.h"
#include "pydocs/geom_docs.h"

namespace indexing {

  struct Index {
    int i; // row
    int j; // col

    const std::string to_string() const {
      return "Index(" + std::to_string(i) + ", " + std::to_string(j) + ")";
    }

    const std::string to_yaml() const {
      return "{i: " + std::to_string(i) + ", j: " + std::to_string(j) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const Index& rc);
    friend bool operator==(const Index& lhs, const Index& rhs);
    friend bool operator!=(const Index& lhs, const Index& rhs);
    friend bool operator==(const Index& lhs, const std::tuple<int, int>& rhs);
    friend bool operator!=(const Index& lhs, const std::tuple<int, int>& rhs);
  };

  std::ostream& operator<<(std::ostream& os, const Index& rc) {
    os << rc.to_string();
    return os;
  }

  bool operator==(const Index& lhs, const Index& rhs){
    return (lhs.i == rhs.i) && (lhs.j == rhs.j);
  }

  bool operator!=(const Index& lhs, const Index& rhs){
    return !operator==(lhs, rhs);
  }

  bool operator==(const Index& lhs, const std::tuple<int, int>& rhs){
    return std::tie(lhs.i, lhs.j) == rhs;
  }

  bool operator!=(const Index& lhs, const std::tuple<int, int>& rhs){
    return !operator==(lhs, rhs);
  }


  struct Point {
    float x; // col, j
    float y; // row, i

    const Index to_index() const {
      return {(int)(floor(y - 0.5f) + 0.5f), (int)(floor(x - 0.5f) + 0.5f)};
    }

    const std::string to_string() const {
      return "Point(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }

    const std::string to_yaml() const {
      return "{x: " + std::to_string(x) + ", y: " + std::to_string(y) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& rc);
    friend bool operator==(const Point& lhs, const Point& rhs);
    friend bool operator!=(const Point& lhs, const Point& rhs);
  };

  std::ostream& operator<<(std::ostream& os, const Point& rc) {
    os << rc.to_string();
    return os;
  }

  bool operator==(const Point& lhs, const Point& rhs){
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
  }

  bool operator!=(const Point& lhs, const Point& rhs){
    return !operator==(lhs, rhs);
  }

  bool operator==(const Point& lhs, const std::tuple<float, float>& rhs){
    return std::tie(lhs.x, lhs.y) == rhs;
  }

  bool operator!=(const Point& lhs, const std::tuple<float, float>& rhs){
    return !operator==(lhs, rhs);
  }


  // A rectangle that also contains it's corner's origin Index with respect to
  // a second origin. Most commonly the corner of another, larger, rectangle -
  // f.e. when pasting stamps at the edge of an image into a new array. Usable
  // as a regular rectangle.
  struct Rectangle {
    Index corner;
    Index anchor;
    unsigned width;
    unsigned height;

    Rectangle(Index corner_idx, Index anchor_idx, unsigned w, unsigned h)
      : corner(corner_idx), anchor(anchor_idx), width(w), height(h) {}

    Rectangle(Index corner_idx, unsigned w, unsigned h)
      : corner(corner_idx), anchor({0, 0}), width(w), height(h) {}

    const std::string to_string() const {
      return "Rectangle(corner:" + corner.to_string() + ", anchor:" + anchor.to_string() +
        ", width: " + std::to_string(width) + ", height: " + std::to_string(height) + ")";
    }

    const std::string to_yaml() const {
      return "{corner: " + corner.to_yaml() + ", anchor: " + anchor.to_yaml() +
        ", width: " + std::to_string(width) + ", height: " + std::to_string(height) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const Rectangle& rc);
    friend bool operator==(const Rectangle& lhs, const Rectangle& rhs);
    friend bool operator!=(const Rectangle& lhs, const Rectangle& rhs);

  };

  std::ostream& operator<<(std::ostream& os, const Rectangle& rc) {
    os << rc.to_string();
    return os;
  }

  bool operator==(const Rectangle& lhs, const Rectangle& rhs){
    return std::tie(lhs.corner, lhs.anchor, lhs.width, lhs.height) ==
      std::tie(rhs.corner, rhs.anchor, rhs.width, rhs.height);
  }

  bool operator!=(const Rectangle& lhs, const Rectangle& rhs){
    return !operator==(lhs, rhs);
  }


  // return an interval [val-r, val+r] pinned to [0, width] range
  inline std::tuple<int, int, int> centered_range(int val, const int r, const int width) {
    // pin start to the [0, width] range
    int start = std::max(0, val - r);
    start = std::min(start, width);

    // pin end to the [0, width]
    int end = std::max(0, val + r);
    end = std::min(width, val + r);

    // range is inclusive of the first element, and can not be longer than start
    // minus max range
    int length = end - start + 1;
    length = std::min(length, width - start);

    return std::make_tuple(start, end, length);
  }


  // get an Eigen block coordinates (top-left, height, width) anchored inside a
  // square matrix of dimensions 2*r+1 (anchor top-left).
  inline Rectangle anchored_block(const Index& idx, const int r,
                                  const unsigned width, const unsigned height) {
    auto [top, bot, rangei] = centered_range(idx.i, r, height);
    auto [left, right, rangej] = centered_range(idx.j, r, width);
    assertm(rangei > 0, "Selected block lies outside of the image limits.");
    assertm(rangej > 0, "Selected block lies outside of the image limits.");

    int anchor_top = std::max(r - idx.i, 0);
    int anchor_left = std::max(r - idx.j, 0);

    // now it's safe to cast ranges to unsigned
    // note that rangej and rangei flip positions because width x height
    return {{top, left}, {anchor_top, anchor_left}, (unsigned)rangej, (unsigned)rangei};
  }


  // returns top-right-bot-left (clockwise) corners around an Index.
  inline auto manhattan_neighbors(const Index& idx,
                                  const unsigned width, const unsigned height) {
    std::array<std::optional<Index>, 4> idxs;

    // top bot
    if (idx.j >= 0 && idx.j < width) {
      if (idx.i-1 >= 0 && idx.i-1 < height) idxs[0] = {idx.i-1, idx.j};
      if (idx.i+1 >= 0 && idx.i+1 < height) idxs[2] = {idx.i+1, idx.j};
    }

    // right left
    if (idx.i >= 0 && idx.i < height) {
      if (idx.j+1 >= 0 && idx.j+1 < width) idxs[1] = {idx.i, idx.j+1};
      if (idx.j-1 >= 0 && idx.j-1 < width) idxs[3] = {idx.i, idx.j-1};
    }
    return idxs;
  }


  // Note the distinct contextual distinction between manhattan neighborhood of
  // Index and Point. Point returns closes **pixel indices** that are neighbors.
  // This includes the pixel the Point is residing within. This is not the case
  // for Index, which will never return itself as a neighbor.
  inline auto manhattan_neighbors(const Point& p,
                                  const unsigned width, const unsigned height) {
    std::array<std::optional<Index>, 4> idxs;

    // The index in which the point resides.
    // Almost always top-left corner, except when
    // point is negative or (0, 0)
    auto idx = p.to_index();

    // top-left bot-right
    if (idx.j >= 0 && idx.j < width) {
      if (idx.i >= 0 && idx.i < height) idxs[0] = {idx.i, idx.j};
      if (idx.i+1 >= 0 && idx.i+1 < height) idxs[3] = {idx.i+1, idx.j};
    }

    // bot-right
    if (idx.i >= 0 && idx.i < height)
      if (idx.j+1 >= 0 && idx.j+1 < width) idxs[1] = {idx.i, idx.j+1};

    // bot-right
    if ((idx.i+1 >= 0 && idx.i+1 < width) && (idx.j+1 >= 0 && idx.j+1 < width))
      idxs[2] = {idx.i+1, idx.j+1};

    return idxs;
  }


#ifdef Py_PYTHON_H
  static void index_bindings(py::module& m) {
    PYBIND11_NUMPY_DTYPE(Index, i, j);
    py::class_<Index>(m, "Index", pydocs::DOC_Index)
      .def(py::init<int, int>())
      .def(py::init<float, float>()) // floor the values explicitly?
      .def_readwrite("i", &Index::i)
      .def_readwrite("j", &Index::j)
      .def("to_yaml", &Index::to_yaml, pydocs::DOC_Index_to_yaml)
      .def("__array__", [](Index& obj){
        py::array_t<Index> arr = py::array_t<Index>({1, });
        py::buffer_info info = arr.request();
        Index* ptr = static_cast<Index*>(info.ptr);
        ptr[0] = obj;
        return arr;
      })
      .def("__repr__", &Index::to_string)
      .def("__str__", &Index::to_string)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self == std::tuple<int, int>())
      .def(py::self != std::tuple<int, int>());
  }


  static void point_bindings(py::module& m) {
    PYBIND11_NUMPY_DTYPE(Point, x, y);
    py::class_<Point>(m, "Point", pydocs::DOC_Point)
      .def(py::init<float, float>())
      .def_readwrite("x", &Point::x)
      .def_readwrite("y", &Point::y)
      .def("to_index", &Point::to_index, pydocs::DOC_Point_to_index)
      .def("to_yaml", &Point::to_yaml, pydocs::DOC_Point_to_yaml)
      .def("__array__", [](Point& obj){
        py::array_t<Point> arr = py::array_t<Point>({1, });
        py::buffer_info info = arr.request();
        Point* ptr = static_cast<Point*>(info.ptr);
        ptr[0] = obj;
        return arr;
      })
      .def("__repr__", &Point::to_string)
      .def("__str__", &Point::to_string)
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self)
      .def(py::self == std::tuple<float, float>())
      .def(py::self != std::tuple<float, float>());
  }


  static void rectangle_bindings(py::module& m) {
    PYBIND11_NUMPY_DTYPE(Rectangle, corner, anchor, width, height);
    py::class_<Rectangle>(m, "Rectangle", pydocs::DOC_Rectangle)
      .def(py::init<Index, unsigned, unsigned>())
      .def(py::init<Index, Index, unsigned, unsigned>())
      .def(py::init([](std::pair<int, int> corner, std::pair<int, int> anchor,
                       unsigned width, unsigned height) {
        return Rectangle{
          {corner.first, corner.second}, {anchor.first, anchor.second}, width, height};
      }))
      .def(py::init([](std::pair<int, int> corner, unsigned height, unsigned width) {
        return Rectangle{{corner.first, corner.second},  height, width};
      }))
      .def_readwrite("width", &Rectangle::width)
      .def_readwrite("height", &Rectangle::height)
      .def_readwrite("corner", &Rectangle::corner)
      .def_readwrite("anchor", &Rectangle::anchor)
      .def_property(
                    "i",
                    /*get*/ [](Rectangle& rect) { return rect.corner.i; },
                    /*set*/ [](Rectangle& rect, int value) { rect.corner.i = value; })
      .def_property(
                    "j",
                    /*get*/ [](Rectangle& rect) { return rect.corner.j; },
                    /*set*/ [](Rectangle& rect, int value) { rect.corner.j = value; })
      .def("to_yaml", &Rectangle::to_yaml, pydocs::DOC_Rectangle_to_yaml)
      .def("__array__", [](Rectangle& obj){
        py::array_t<Rectangle> arr = py::array_t<Rectangle>({1, });
        py::buffer_info info = arr.request();
        Rectangle* ptr = static_cast<Rectangle*>(info.ptr);
        ptr[0] = obj;
        return arr;
      })
      .def("__repr__", &Rectangle::to_string)
      .def("__str__", &Rectangle::to_string)
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self);
  }


  static void geom_functions(py::module& m) {
    m.def("centered_range", &centered_range, pydocs::DOC_centered_range);

    // numpy.shape returns (nrows, ncols), i.e. (height, width)
    // so we need to flip the shape order as input to anchored_block
    m.def("anchored_block",
          [](std::pair<int, int> idx, const int r, std::pair<int, int> shape){
            return anchored_block({idx.first, idx.second}, r, shape.second, shape.first);
          }, pydocs::DOC_anchored_block);

    // Safe to cast to int as "ij" implies user is using indices, i.e. ints.
    // CPP can be so odd, why not return a 1 or, you know... a bool? Mostly for
    // testing purposes.
    m.def("manhattan_neighbors", [](const std::pair<float, float> coords,
                                    const std::pair<unsigned, unsigned> shape,
                                    const std::string indexing){
      if (indexing.compare("ij") == 0)
        return manhattan_neighbors(Index{(int)coords.first, (int)coords.second},
                                   shape.second, shape.first);
      else if (indexing.compare("xy") == 0)
        return manhattan_neighbors(Point{coords.first, coords.second},
                                   shape.second, shape.first);
      else
        throw std::domain_error("Expected 'ij' or 'xy' got " + indexing + " instead.");
    }, pydocs::DOC_manhattan_neighbors);
  }
#endif  // Py_PYTHON_H
}  // namespace indexing
#endif  // GEOM_H
