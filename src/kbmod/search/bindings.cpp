#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>  // still required for PSF.h
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "logging.h"
#include "common.h"

#include "cpu_search_algorithms.cpp"
#include "stack_search.cpp"
#include "kernel_testing_helpers.cpp"
#include "psi_phi_array.cpp"
#include "debug_timer.cpp"
#include "trajectory_list.cpp"
#include "image_utils_cpp.cpp"

PYBIND11_MODULE(search, m) {
    m.attr("KB_NO_DATA") = pybind11::float_(search::NO_DATA);
    m.attr("HAS_GPU") = pybind11::bool_(search::HAVE_GPU);
    m.attr("HAS_OMP") = pybind11::bool_(search::HAVE_OMP);
    py::enum_<search::StampType>(m, "StampType")
            .value("STAMP_SUM", search::StampType::STAMP_SUM)
            .value("STAMP_MEAN", search::StampType::STAMP_MEAN)
            .value("STAMP_MEDIAN", search::StampType::STAMP_MEDIAN)
            .value("STAMP_VAR_WEIGHTED", search::StampType::STAMP_VAR_WEIGHTED)
            .export_values();
    logging::logging_bindings(m);
    search::cpu_search_algorithms_bindings(m);
    search::stack_search_bindings(m);
    search::trajectory_bindings(m);
    search::search_parameters_bindings(m);
    search::psi_phi_array_binding(m);
    search::debug_timer_binding(m);
    search::trajectory_list_binding(m);
    // Helper function from common.h
    m.def("pixel_value_valid", &search::pixel_value_valid);
    // Functions from kernel_testing_helpers.cpp
    search::kernel_helper_bindings(m);
    search::image_utils_cpp(m);
}
