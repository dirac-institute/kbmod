#ifndef GEOM_H_
#define GEOM_H_

#include <iostream>
#include <optional>
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
      return "Index(" + std::to_string(i) + ", " + std::to_string(j) +")";
    }

    const std::string to_yaml() const {
      return "{x: " + std::to_string(i) + " y: " + std::to_string(j) + "}";
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
      return {(int)floor(y), (int)floor(x)};
    }

    const std::string to_string() const {
      return "Point(" + std::to_string(x) + ", " + std::to_string(y) +")";
    }

    const std::string to_yaml() const {
      return "{x: " + std::to_string(x) + " y: " + std::to_string(y) + "}";
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& rc);
  };

  std::ostream& operator<<(std::ostream& os, const Point& rc){
    os << rc.to_string();
    return os;
  }


  struct Rectangle{
    Index idx;
    unsigned width;
    unsigned height;

    const std::string to_string() const {
      return "Rectangle(" + idx.to_yaml() + ", dx: " + std::to_string(width) + \
        ", dy: " + std::to_string(height) + ")";
    }

    const std::string to_yaml() const {
      return "{idx: " + idx.to_yaml() +      \
        " width: " + std::to_string(width) + \
        " height: " + std::to_string(height);
    }

    friend std::ostream& operator<<(std::ostream& os, const Rectangle& rc);
  };

  std::ostream& operator<<(std::ostream& os, const Rectangle& rc){
    os << rc.to_string();
    return os;
  }


  inline Rectangle centered_block(const Index& idx,
                                  const int r,
                                  const unsigned width,
                                  const unsigned height) {
    int left_x = ((idx.i-r >= 0) && (idx.i-r < width)) ? idx.i-r : idx.i;
    int right_x = ((idx.i+r >= 0) && (idx.i+r < width)) ? idx.i+r : width - idx.i;
    int top_y = ((idx.j-r >= 0) && (idx.j-r < height)) ? idx.j-r : idx.j;
    int bot_y = ((idx.j+r >= 0) && (idx.j+r < height)) ? idx.j+r : height - idx.i;
    assertm(bot_y - top_y > 0, "Rectangle has negative height!");
    assertm(right_x - left_x > 0, "Rectangle has negative width!");
    unsigned dx = right_x - left_x + 1;
    unsigned dy = bot_y - top_y + 1;
    return {{top_y, left_x}, dy, dx};
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

    // top-left bot-left
    if (idx.i >= 0 && idx.i<width){
      if (idx.j >= 0 && idx.j<height)
        idxs[0] = {idx.j, idx.i};
      if (idx.j+1 >= 0 && idx.j+1<height)
        idxs[1] = {idx.j+1, idx.i};
    }

    // top-right
    if (idx.j >= 0 && idx.j<height)
      if (idx.i+1 >= 0 && idx.i+1<width)
        idxs[3] = {idx.j, idx.i+1};

    // bot-right
    if ((idx.j+1 >= 0 && idx.j+1<width) &&
        (idx.i+1 >= 0 && idx.i+1<width))
      idxs[2] = {idx.j+1, idx.i+1};

    return idxs;
  }


  /*#ifndef NDEBUG
  // these are helper functions not used in the code, but help with debugging
  inline std::vector<Index> all_neighbors(const Index& idx,
  const unsigned width,
  const unsigned height) {
  auto res = manhattan_neighbors(idx, width, height);

  // top-left
  if ((idx.i-1 >= 0 && idx.i-1<width ? true : false) &&
  (idx.j-1 >= 0 && idx.j-1<height ? true : false))
  idxs.push_back(Index(idx.i-1, idx.j-1));

  // top-right
  if ((idx.i+1 >= 0 && idx.i+1<width ? true : false) &&
  (idx.j-1 >= 0 && idx.j-1<height ? true : false))
  idxs.push_back(Index(idx.i+1, idx.j-1));

  // bot left
  if ((idx.i-1 >= 0 && idx.i-1<width ? true : false) &&
  (idx.j+1 >= 0 && idx.j+1<height ? true : false))
  idxs.push_back(Index(idx.i-1, idx.j+1));

  // bot right
  if ((idx.i+1 >= 0 && idx.i+1<width ? true : false) &&
  (idx.j+1 >= 0 && idx.j+1<height ? true : false))
  idxs.push_back(Index(idx.i+1, idx.j+11));

  return idxs;
  }
  #endif // NDEBUG
  */

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
      .def("to_yaml", &Point::to_yaml)
      .def("__repr__", &Point::to_string)
      .def("__str__", &Point::to_string);
  }

  static void rectangle_bindings(py::module &m) {
    py::class_<Rectangle>(m, "Rectangle")
      .def(py::init<Index, unsigned, unsigned>())
      .def(py::init( [](int i, int j, unsigned width, unsigned height)
      { return Rectangle{{i, j}, width, height }; } ))
      .def(py::init<int, int, float, float>())
      .def_readwrite("width", &Rectangle::width)
      .def_readwrite("height", &Rectangle::height)
      .def_property("i",
                    /*get*/ [](Rectangle& rect) { return rect.idx.i;},
                    /*set*/ [](Rectangle& rect, int value) { rect.idx.i = value; })
      .def_property("j",
                    /*get*/ [](Rectangle& rect) { return rect.idx.j;},
                    /*set*/ [](Rectangle& rect, int value) { rect.idx.j = value; })
      .def("to_yaml", &Rectangle::to_yaml)
      .def("__repr__", &Rectangle::to_string)
      .def("__str__", &Rectangle::to_string);
  }
#endif // Py_PYTHON_H

} // namespace indexing

#endif // GEOM_H
