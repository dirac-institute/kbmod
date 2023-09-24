#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "psf.cpp"
#include "raw_image.cpp"
#include "layered_image.cpp"
#include "image_stack.cpp"
#include "stack_search.cpp"
#include "Filtering.cpp"
#include "common.h"


using bc = search::BaryCorrection;
using pp = search::PixelPos;
using std::to_string;


PYBIND11_MODULE(search, m) {
  m.attr("KB_NO_DATA") = pybind11::float_(search::NO_DATA);
  m.attr("HAS_GPU") = pybind11::bool_(search::HAVE_GPU);
  py::enum_<search::StampType>(m, "StampType")
    .value("STAMP_SUM", search::StampType::STAMP_SUM)
    .value("STAMP_MEAN", search::StampType::STAMP_MEAN)
    .value("STAMP_MEDIAN", search::StampType::STAMP_MEDIAN)
    .export_values();
  search::psf_bindings(m);
  search::raw_image_bindings(m);
  search::layered_image_bindings(m);
  search::image_stack_bindings(m);
  search::stack_search_bindings(m);
  search::trajectory_bindings(m);
  search::pixel_pos_bindings(m);
  m.def("create_median_image", &search::create_median_image);
  m.def("create_summed_image", &search::create_summed_image);
  m.def("create_mean_image", &search::create_mean_image);
  py::class_<search::ImageMoments>(m, "image_moments")
    .def(py::init<>())
    .def_readwrite("m00", &search::ImageMoments::m00)
    .def_readwrite("m01", &search::ImageMoments::m01)
    .def_readwrite("m10", &search::ImageMoments::m10)
    .def_readwrite("m11", &search::ImageMoments::m11)
    .def_readwrite("m02", &search::ImageMoments::m02)
    .def_readwrite("m20", &search::ImageMoments::m20);
  py::class_<search::StampParameters>(m, "stamp_parameters")
    .def(py::init<>())
    .def_readwrite("radius", &search::StampParameters::radius)
    .def_readwrite("stamp_type", &search::StampParameters::stamp_type)
    .def_readwrite("do_filtering", &search::StampParameters::do_filtering)
    .def_readwrite("center_thresh", &search::StampParameters::center_thresh)
    .def_readwrite("peak_offset_x", &search::StampParameters::peak_offset_x)
    .def_readwrite("peak_offset_y", &search::StampParameters::peak_offset_y)
    .def_readwrite("m01", &search::StampParameters::m01_limit)
    .def_readwrite("m10", &search::StampParameters::m10_limit)
    .def_readwrite("m11", &search::StampParameters::m11_limit)
    .def_readwrite("m02", &search::StampParameters::m02_limit)
    .def_readwrite("m20", &search::StampParameters::m20_limit);
  py::class_<bc>(m, "BaryCorrection")
    .def(py::init<>())
    .def_readwrite("dx", &bc::dx)
    .def_readwrite("dxdx", &bc::dxdx)
    .def_readwrite("dxdy", &bc::dxdy)
    .def_readwrite("dy", &bc::dy)
    .def_readwrite("dydx", &bc::dydx)
    .def_readwrite("dydy", &bc::dydy)
    .def("__repr__", [](const bc &b) {
      return "dx = " + to_string(b.dx) + " + " + to_string(b.dxdx) + " x + " + to_string(b.dxdy) +
        " y; " + " dy = " + to_string(b.dy) + " + " + to_string(b.dydx) + " x + " +
        to_string(b.dydy) + " y";
    });
  // Functions from Filtering.cpp
  m.def("sigmag_filtered_indices", &search::sigmaGFilteredIndices);
  m.def("calculate_likelihood_psi_phi", &search::calculateLikelihoodFromPsiPhi);
}
