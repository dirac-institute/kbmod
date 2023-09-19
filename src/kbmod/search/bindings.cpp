#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "KBMOSearch.cpp"
#include "Filtering.cpp"

namespace py = pybind11;

using tj = trajectory;
using is = image_base::ImageStack;
using ri = image_base::RawImage;
using ks = search::KBMOSearch;

using std::to_string;

PYBIND11_MODULE(search, m) {
    py::enum_<StampType>(m, "StampType")
            .value("STAMP_SUM", StampType::STAMP_SUM)
            .value("STAMP_MEAN", StampType::STAMP_MEAN)
            .value("STAMP_MEDIAN", StampType::STAMP_MEDIAN)
            .export_values();
    py::class_<trajectory>(m, "trajectory", R"pbdoc(
            A trajectory structure holding basic information about potential results.
            )pbdoc")
            .def(py::init<>())
            .def_readwrite("x_v", &trajectory::xVel)
            .def_readwrite("y_v", &trajectory::yVel)
            .def_readwrite("lh", &trajectory::lh)
            .def_readwrite("flux", &trajectory::flux)
            .def_readwrite("x", &trajectory::x)
            .def_readwrite("y", &trajectory::y)
            .def_readwrite("obs_count", &trajectory::obsCount)
            .def("__repr__",
                 [](const trajectory &t) {
                     return "lh: " + to_string(t.lh) + " flux: " + to_string(t.flux) +
                            " x: " + to_string(t.x) + " y: " + to_string(t.y) + " x_v: " + to_string(t.xVel) +
                            " y_v: " + to_string(t.yVel) + " obs_count: " + to_string(t.obsCount);
                 })
            .def(py::pickle(
                    [](const trajectory &p) {  // __getstate__
                        return py::make_tuple(p.xVel, p.yVel, p.lh, p.flux, p.x, p.y, p.obsCount);
                    },
                    [](py::tuple t) {  // __setstate__
                        if (t.size() != 7) throw std::runtime_error("Invalid state!");
                        trajectory trj = {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(),
                                          t[3].cast<float>(), t[4].cast<short>(), t[5].cast<short>(),
                                          t[6].cast<short>()};
                        return trj;
                    }));
    py::class_<stampParameters>(m, "stamp_parameters")
            .def(py::init<>())
            .def_readwrite("radius", &stampParameters::radius)
            .def_readwrite("stamp_type", &stampParameters::stamp_type)
            .def_readwrite("do_filtering", &stampParameters::do_filtering)
            .def_readwrite("center_thresh", &stampParameters::center_thresh)
            .def_readwrite("peak_offset_x", &stampParameters::peak_offset_x)
            .def_readwrite("peak_offset_y", &stampParameters::peak_offset_y)
            .def_readwrite("m01", &stampParameters::m01_limit)
            .def_readwrite("m10", &stampParameters::m10_limit)
            .def_readwrite("m11", &stampParameters::m11_limit)
            .def_readwrite("m02", &stampParameters::m02_limit)
            .def_readwrite("m20", &stampParameters::m20_limit);
    py::class_<ks>(m, "stack_search")
            .def(py::init<is &>())
            .def("save_psi_phi", &ks::savePsiPhi)
            .def("search", &ks::search)
            .def("enable_gpu_sigmag_filter", &ks::enableGPUSigmaGFilter)
            .def("enable_gpu_encoding", &ks::enableGPUEncoding)
            .def("enable_corr", &ks::enableCorr)
            .def("set_start_bounds_x", &ks::setStartBoundsX)
            .def("set_start_bounds_y", &ks::setStartBoundsY)
            .def("set_debug", &ks::setDebug)
            .def("filter_min_obs", &ks::filterResults)
            .def("get_num_images", &ks::numImages)
            .def("get_image_stack", &ks::getImageStack)
            // Science Stamp Functions
            .def("science_viz_stamps", &ks::scienceStampsForViz)
            .def("median_sci_stamp", &ks::medianScienceStamp)
            .def("mean_sci_stamp", &ks::meanScienceStamp)
            .def("summed_sci_stamp", &ks::summedScienceStamp)
            .def("coadded_stamps",
                 (std::vector<ri>(ks::*)(std::vector<tj> &, std::vector<std::vector<bool>> &,
                                         const stampParameters &, bool)) &
                         ks::coaddedScienceStamps)
            // For testing
            .def("filter_stamp", &ks::filterStamp)
            .def("get_traj_pos", &ks::getTrajPos)
            .def("get_mult_traj_pos", &ks::getMultTrajPos)
            .def("psi_curves", (std::vector<float>(ks::*)(tj &)) & ks::psiCurves)
            .def("phi_curves", (std::vector<float>(ks::*)(tj &)) & ks::phiCurves)
            .def("prepare_psi_phi", &ks::preparePsiPhi)
            .def("get_psi_images", &ks::getPsiImages)
            .def("get_phi_images", &ks::getPhiImages)
            .def("get_results", &ks::getResults)
            .def("set_results", &ks::setResults);

    // Functions from Filtering.cpp
    m.def("sigmag_filtered_indices", &search::sigmaGFilteredIndices);
    m.def("calculate_likelihood_psi_phi", &search::calculateLikelihoodFromPsiPhi);
}
