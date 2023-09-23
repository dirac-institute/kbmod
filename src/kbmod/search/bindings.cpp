#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "psf.cpp"
#include "raw_image.cpp"
#include "layered_image.cpp"
#include "ImageStack.cpp"
#include "KBMOSearch.cpp"
#include "Filtering.cpp"


using pf = search::PSF;
using ri = search::RawImage;
using li = search::LayeredImage;
using is = search::ImageStack;
using ks = search::KBMOSearch;
using tj = search::trajectory;
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
    m.def("create_median_image", &search::create_median_image);
    m.def("create_summed_image", &search::create_summed_image);
    m.def("create_mean_image", &search::create_mean_image);
    py::class_<is>(m, "image_stack")
            .def(py::init<std::vector<std::string>, std::vector<pf>>())
            .def(py::init<std::vector<li>>())
            .def("get_images", &is::get_images)
            .def("get_single_image", &is::get_single_image)
            .def("set_single_image", &is::set_single_image)
            .def("get_times", &is::get_times)
            .def("set_times", &is::set_times)
            .def("img_count", &is::img_count)
            .def("apply_mask_flags", &is::apply_mask_flags)
            .def("apply_mask_threshold", &is::apply_mask_threshold)
            .def("apply_global_mask", &is::apply_global_mask)
            .def("grow_mask", &is::grow_mask)
            .def("save_global_mask", &is::save_global_mask)
            .def("save_images", &is::save_images)
            .def("get_global_mask", &is::get_global_mask)
            .def("convolve_psf", &is::convolve_psf)
            .def("get_width", &is::get_width)
            .def("get_height", &is::getHeight)
            .def("get_npixels", &is::get_npixels);
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
                                         const search::StampParameters &, bool)) &
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
    py::class_<tj>(m, "trajectory", R"pbdoc(
            A trajectory structure holding basic information about potential results.
            )pbdoc")
            .def(py::init<>())
            .def_readwrite("x_v", &tj::x_vel)
            .def_readwrite("y_v", &tj::y_vel)
            .def_readwrite("lh", &tj::lh)
            .def_readwrite("flux", &tj::flux)
            .def_readwrite("x", &tj::x)
            .def_readwrite("y", &tj::y)
            .def_readwrite("obs_count", &tj::obs_count)
            .def("__repr__",
                 [](const tj &t) {
                     return "lh: " + to_string(t.lh) + " flux: " + to_string(t.flux) +
                            " x: " + to_string(t.x) + " y: " + to_string(t.y) +
                            " x_v: " + to_string(t.x_vel) + " y_v: " + to_string(t.y_vel) +
                            " obs_count: " + to_string(t.obs_count);
                 })
            .def(py::pickle(
                    [](const tj &p) {  // __getstate__
                        return py::make_tuple(p.x_vel, p.y_vel, p.lh, p.flux, p.x, p.y, p.obs_count);
                    },
                    [](py::tuple t) {  // __setstate__
                        if (t.size() != 7) throw std::runtime_error("Invalid state!");
                        tj trj = {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(),
                                  t[3].cast<float>(), t[4].cast<short>(), t[5].cast<short>(),
                                  t[6].cast<short>()};
                        return trj;
                    }));
    py::class_<pp>(m, "pixel_pos")
            .def(py::init<>())
            .def_readwrite("x", &pp::x)
            .def_readwrite("y", &pp::y)
            .def("__repr__", [](const pp &p) { return "x: " + to_string(p.x) + " y: " + to_string(p.y); });
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
