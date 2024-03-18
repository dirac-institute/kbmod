#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>  // still required for PSF.h
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "logging.h"
#include "common.h"
#include "geom.h"

#include "psf.cpp"
#include "raw_image.cpp"
#include "layered_image.cpp"
#include "image_stack.cpp"
#include "stack_search.cpp"
#include "stamp_creator.cpp"
#include "kernel_testing_helpers.cpp"
#include "psi_phi_array.cpp"
#include "debug_timer.cpp"
#include "trajectory_list.cpp"

PYBIND11_MODULE(search, m) {
    m.attr("KB_NO_DATA") = pybind11::float_(search::NO_DATA);
    m.attr("HAS_GPU") = pybind11::bool_(search::HAVE_GPU);
    py::enum_<search::StampType>(m, "StampType")
            .value("STAMP_SUM", search::StampType::STAMP_SUM)
            .value("STAMP_MEAN", search::StampType::STAMP_MEAN)
            .value("STAMP_MEDIAN", search::StampType::STAMP_MEDIAN)
            .export_values();
    logging::logging_bindings(m);
    indexing::index_bindings(m);
    indexing::point_bindings(m);
    indexing::rectangle_bindings(m);
    indexing::geom_functions(m);
    search::psf_bindings(m);
    search::raw_image_bindings(m);
    search::layered_image_bindings(m);
    search::image_stack_bindings(m);
    search::stack_search_bindings(m);
    search::stamp_creator_bindings(m);
    search::trajectory_bindings(m);
    search::image_moments_bindings(m);
    search::stamp_parameters_bindings(m);
    search::psi_phi_array_binding(m);
    search::debug_timer_binding(m);
    search::trajectory_list_binding(m);
    // Helper function from common.h
    m.def("pixel_value_valid", &search::pixel_value_valid);
    // Functions from raw_image.cpp
    m.def("create_median_image", &search::create_median_image);
    m.def("create_summed_image", &search::create_summed_image);
    m.def("create_mean_image", &search::create_mean_image);
    // Functions from kernel_testing_helpers.cpp
    m.def("sigmag_filtered_indices", &search::sigmaGFilteredIndices);
}
