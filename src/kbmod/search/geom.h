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
//#include "pydocs/geom_docs.h"


namespace indexing {

  struct Index{
    int j;
    int i;

    const std::string to_string() const {
      return "Index(" + std::to_string(j) + ", " + std::to_string(i) +")";
    }

    const std::string to_yaml() const {
      return "{j: " + std::to_string(j) + " i: " + std::to_string(i) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const Index& rc);
  };

  std::ostream& operator<<(std::ostream& os, const Index& rc){
    os << rc.to_string();
    return os;
  }


  struct Point{
    float y;
    float x;

    const Index to_index() const {
      return {(int)(floor(y-0.5f)+0.5f), (int)(floor(x-0.5f)+0.5f)};
    }

    const std::string to_string() const {
      return "Point(" + std::to_string(y) + ", " + std::to_string(x) +")";
    }

    const std::string to_yaml() const {
      return "{x: " + std::to_string(y) + " y: " + std::to_string(x) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& rc);
  };

  std::ostream& operator<<(std::ostream& os, const Point& rc){
    os << rc.to_string();
    return os;
  }


  // A rectangle that also contains it's corner's origin Index with respect to
  // another, larger rectangle.
  struct AnchoredRectangle{
    Index corner;
    Index anchor;

    unsigned height;
    unsigned width;

    const std::string to_string() const {
      return "AnchoredRectangle(corner:" + corner.to_string() +
        ", anchor:" + anchor.to_string() +
        ", height: " + std::to_string(height) +
        ", width: " + std::to_string(width) + ")";
    }

    const std::string to_yaml() const {
      return "{corner: " + corner.to_yaml() +
        " anchor: " + anchor.to_yaml() +
        " height: " + std::to_string(height) +
        " width: " + std::to_string(width) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const AnchoredRectangle& rc);
  };

  std::ostream& operator<<(std::ostream& os, const AnchoredRectangle& rc){
    os << rc.to_string();
    return os;
  }


  // Given an index component `idx_val`, and a radius `r` around it, returns the
  // start and end index components, and length of the range, that fit in the
  // [0, max_range] limits.
  inline std::tuple<int, int, int> centered_range(int idx_val, const int r,
                                                  const int max_range) {
    // pin start to the []0, max_range] range
    int start = std::max(0, idx_val-r);
    start = std::min(start, max_range);

    // pin end to the [0, max_range]
    int end = std::max(0, idx_val+r);
    end = std::min(max_range, idx_val+r);

    // range is inclusive of the first element, and can not be longer than start
    // minus max range
    int length = end-start+1;
    length = std::min(length, max_range-start);

    return std::make_tuple(start, end, length);
  }


  // get an Eigen block coordinates (top-left, height, width) anchored inside a
  // square matrix of dimensions 2*r+1 (anchor top-left).
  inline AnchoredRectangle anchored_block(const Index& idx,
                                          const int r,
                                          const unsigned height,
                                          const unsigned width) {
    auto [top, bot, rangey] = centered_range(idx.j, r, height);
    auto [left, right, rangex] = centered_range(idx.i, r, width);
    assertm(rangey > 0, "Selected block lies outside of the image limits.");
    assertm(rangex > 0, "Selected block lies outside of the image limits.");

    int anchor_top = std::max(r-idx.j, 0);
    int anchor_left = std::max(r-idx.i, 0);

    // now it's safe to cast ranges to unsigned
    return {{top, left}, {anchor_top, anchor_left}, (unsigned)rangey, (unsigned)rangex};
  }


  inline auto manhattan_neighbors(const Index& idx,
                                  const unsigned width,
                                  const unsigned height) {
    std::array<std::optional<Index>, 4> idxs;

    // top bot
    if (idx.i >= 0 && idx.i<width){
      if (idx.j-1 >= 0 && idx.j-1<height)
        idxs[0] = {idx.j-1, idx.i};
      if (idx.j+1 >= 0 && idx.j+1<height)
        idxs[1] = {idx.j+1, idx.i};
    }

    // left right
    if (idx.j >= 0 && idx.j<height){
      if (idx.i-1 >= 0 && idx.i-1<width)
        idxs[2] = {idx.j, idx.i-1};
      if (idx.i+1 >= 0 && idx.i+1<width)
        idxs[3] = {idx.j, idx.i+1};
    }
    return idxs;
  }


  // Note the distinct contextual distinction between
  // manhattan neighborhood of Index and Point.
  // Point returns closes **pixel indices** that
  // are neighbors. This includes the pixel the Point is
  // residing within. This is not the case for Index, which
  // will never return itself as a neighbor.
  inline auto manhattan_neighbors(const Point& p,
                                  const unsigned width,
                                  const unsigned height) {
    std::array<std::optional<Index>, 4> idxs;

    // The index in which the point resides.
    // Almost always top-left corner, except when
    // point is negative or (0, 0)
    auto idx = p.to_index();

    // top-left bot-right
    if (idx.i >= 0 && idx.i<width){
      if (idx.j >= 0 && idx.j<height)
        idxs[0] = {idx.j, idx.i};
      if (idx.j+1 >= 0 && idx.j+1<height)
        idxs[1] = {idx.j+1, idx.i};
    }

    // bot-right
    if (idx.j >= 0 && idx.j<height)
      if (idx.i+1 >= 0 && idx.i+1<width)
        idxs[3] = {idx.j, idx.i+1};

    // bot-right
    if ((idx.j+1 >= 0 && idx.j+1<width) &&
        (idx.i+1 >= 0 && idx.i+1<width))
      idxs[2] = {idx.j+1, idx.i+1};

    return idxs;
  }


#ifdef Py_PYTHON_H
  static void index_bindings(py::module &m) {
    py::class_<Index>(m, "Index")
      .def(py::init<int, int>())
      .def(py::init<float, float>())
      .def_readwrite("i", &Index::i)
      .def_readwrite("j", &Index::j)
      .def("to_yaml", &Index::to_yaml)
      .def("__repr__", &Index::to_string)
      .def("__str__", &Index::to_string);
  }

  static void point_bindings(py::module &m) {
    py::class_<Point>(m, "Point")
      .def(py::init<float, float>())
      .def_readwrite("x", &Point::x)
      .def_readwrite("y", &Point::y)
      .def("to_index", &Point::to_index)
      .def("to_yaml", &Point::to_yaml)
      .def("__repr__", &Point::to_string)
      .def("__str__", &Point::to_string);
  }

  static void anchored_rectangle_bindings(py::module &m) {
    py::class_<AnchoredRectangle>(m, "AnchoredRectangle")
      .def(py::init<Index, Index, unsigned, unsigned>())
      .def(py::init( [](std::pair<int, int> corner, std::pair<int, int> anchor, unsigned height, unsigned width) {
        return AnchoredRectangle{{corner.first, corner.second}, {anchor.first, anchor.second}, height, width};
      }))
      .def(py::init( [](std::pair<int, int> corner, unsigned height, unsigned width) {
        return AnchoredRectangle{{corner.first, corner.second}, {0, 0}, height, width};
      }))
      .def(py::init<int, int, float, float>())
      .def_readwrite("width", &AnchoredRectangle::width)
      .def_readwrite("height", &AnchoredRectangle::height)
      .def_readwrite("corner", &AnchoredRectangle::corner)
      .def_readwrite("anchor", &AnchoredRectangle::anchor)
      .def_property("i",
                    /*get*/ [](AnchoredRectangle& rect) { return rect.corner.i;},
                    /*set*/ [](AnchoredRectangle& rect, int value) { rect.corner.i = value; })
      .def_property("j",
                    /*get*/ [](AnchoredRectangle& rect) { return rect.corner.j;},
                    /*set*/ [](AnchoredRectangle& rect, int value) { rect.corner.j = value; })
      .def("to_yaml", &AnchoredRectangle::to_yaml)
      .def("__repr__", &AnchoredRectangle::to_string)
      .def("__str__", &AnchoredRectangle::to_string);
  }
#endif // Py_PYTHON_H

} // namespace indexing

#endif // GEOM_H
