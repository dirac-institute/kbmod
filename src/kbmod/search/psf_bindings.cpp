#ifndef PSF_BINDINGS
#define PSF_BINDINGS

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "psf.h"


namespace py = pybind11;


namespace search {

  static const auto DOC_PSF = R"doc(
  Point Spread Function.

  Parameters
  ----------
  stdev : `float`, optional
      Standard deviation of the Gaussian PSF.
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

  static void psf_bindings_factory(py::module &m) {

    using psf = image_base::PSF;

    py::class_<psf>(m, "PSF", py::buffer_protocol(), DOC_PSF)
      .def_buffer([](psf &m) -> py::buffer_info {
        return py::buffer_info(m.data(), sizeof(float), py::format_descriptor<float>::format(),
                               2, {m.get_dim(), m.get_dim()},
                               {sizeof(float) * m.get_dim(), sizeof(float)});
      })
      .def(py::init<float>())
      .def(py::init<py::array_t<float>>())
      .def(py::init<psf &>())
      .def("set_array", &psf::set_array, DOC_PSF_set_array)
      .def("get_stdev", &psf::get_stdev, "Returns the PSF's standard deviation.")
      .def("get_sum", &psf::get_sum, "Returns the sum of PSFs kernel elements.")
      .def("get_dim", &psf::get_dim, "Returns the PSF kernel dimensions.")
      .def("get_radius", &psf::get_radius, "Returns the radius of the PSF")
      .def("get_size", &psf::get_size, "Returns the number of elements in the PSFs kernel.")
      .def("get_kernel", &psf::get_kernel, "Returns the PSF kernel.")
      .def("get_value", &psf::get_value, "Returns the PSF kernel value at a specific point.")
      .def("square_psf", &psf::square_psf,
           "Squares, raises to the power of two, the elements of the PSF kernel.")
      .def("print_psf", &psf::print, "Pretty-prints the PSF.");
  }

} /* namesapce image_base_bindings */

#endif /* PSF_BINDINGS */
