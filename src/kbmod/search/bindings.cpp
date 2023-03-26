#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "PointSpreadFunc.cpp"
#include "RawImage.cpp"
#include "LayeredImage.cpp"
#include "ImageStack.cpp"
#include "KBMOSearch.cpp"
#include "Filtering.cpp"
#include "TrajectoryUtils.cpp"

namespace py = pybind11;

using pf = search::PointSpreadFunc;
using ri = search::RawImage;
using li = search::LayeredImage;
using is = search::ImageStack;
using ks = search::KBMOSearch;
using tj = search::trajectory;
using bc = search::baryCorrection;
using pp = search::pixelPos;

using std::to_string;

PYBIND11_MODULE(search, m) {
    m.attr("KB_NO_DATA") = pybind11::float_(search::NO_DATA);
    py::enum_<search::StampType>(m, "StampType")
            .value("STAMP_SUM", search::StampType::STAMP_SUM)
            .value("STAMP_MEAN", search::StampType::STAMP_MEAN)
            .value("STAMP_MEDIAN", search::StampType::STAMP_MEDIAN)
            .export_values();
    py::class_<pf>(m, "psf", py::buffer_protocol(), R"pbdoc(
            Point Spread Function.

            Parameters
            ----------
            arg : `float`, `numpy.array` or `psf`
                Given value represents one of:

                * standard deviation of a Gaussian PSF (`float`)
                * kernel representing the PSF (array-like)
                * another `psf` object.


            Notes
            -----
            When instantiated with another `psf` object, returns its copy.

            When instantiated with an array-like object, that array must be
            a square matrix and have an odd number of dimensions
            )pbdoc")
            .def_buffer([](pf &m) -> py::buffer_info {
                return py::buffer_info(m.kernelData(), sizeof(float), py::format_descriptor<float>::format(),
                                       2, {m.getDim(), m.getDim()},
                                       {sizeof(float) * m.getDim(), sizeof(float)});
            })
            .def(py::init<float>())
            .def(py::init<py::array_t<float>>())
            .def(py::init<pf &>())
            .def("set_array", &pf::setArray, R"pbdoc(
            Sets the PSF kernel.

            Parameters
            ----------
            arr : `numpy.array`
                Numpy array representing the PSF.
            )pbdoc")
            .def("get_stdev", &pf::getStdev, "Returns the PSF's standard deviation.")
            .def("get_sum", &pf::getSum, "Returns the sum of PSFs kernel elements.")
            .def("get_dim", &pf::getDim, "Returns the PSF kernel dimensions.")
            .def("get_radius", &pf::getRadius, "Returns the radius of the PSF")
            .def("get_size", &pf::getSize, "Returns the number of elements in the PSFs kernel.")
            .def("get_kernel", &pf::getKernel, "Returns the PSF kernel.")
            .def("square_psf", &pf::squarePSF,
                 "Squares, raises to the power of two, the elements of the PSF kernel.")
            .def("print_psf", &pf::printPSF, "Pretty-prints the PSF.");

    py::class_<ri>(m, "raw_image", py::buffer_protocol())
            .def_buffer([](ri &m) -> py::buffer_info {
                return py::buffer_info(m.getDataRef(), sizeof(float), py::format_descriptor<float>::format(),
                                       2, {m.getHeight(), m.getWidth()},
                                       {sizeof(float) * m.getWidth(), sizeof(float)});
            })
            .def(py::init<int, int>())
            .def(py::init<py::array_t<float>>())
            .def("get_height", &ri::getHeight, "Returns the image's height in pixels.")
            .def("get_width", &ri::getWidth, "Returns the image's width in pixels.")
            .def("get_ppi", &ri::getPPI, "Returns the image's total number of pixels.")
            .def("get_all_pixels", &ri::getPixels, "Returns a list of the images pixels.")
            .def("set_array", &ri::setArray, "Sets all image pixels given an array of values.")
            .def("compute_bounds", &ri::computeBounds, "Returns min and max pixel values.")
            .def("find_peak", &ri::findPeak, "Returns the pixel coordinates of the maximum value.")
            .def("find_central_moments", &ri::findCentralMoments, "Returns the central moments of the image.")
            .def("create_stamp", &ri::createStamp)
            .def("set_pixel", &ri::setPixel, "Set the value of a given pixel.")
            .def("add_pixel", &ri::addToPixel, "Add to the value of a given pixel.")
            .def("apply_mask", &ri::applyMask)
            .def("grow_mask", &ri::growMask)
            .def("pixel_has_data", &ri::pixelHasData,
                 "Returns a Boolean indicating whether the pixel has data.")
            .def("set_all", &ri::setAllPix, "Set all pixel values given an array.")
            .def("get_pixel", &ri::getPixel, "Returns the value of a pixel.")
            .def("get_pixel_interp", &ri::getPixelInterp, "Get the interoplated value of a pixel.")
            .def("convolve", &ri::convolve, "Convolve the image with a PSF.")
            .def("save_fits", &ri::saveToFile, "Save the image to a FITS file.");
    m.def("create_median_image", &search::createMedianImage);
    m.def("create_summed_image", &search::createSummedImage);
    m.def("create_mean_image", &search::createMeanImage);
    py::class_<li>(m, "layered_image")
            .def(py::init<const std::string, pf &>())
            .def(py::init<std::string, int, int, double, float, float, pf &>())
            .def(py::init<std::string, int, int, double, float, float, pf &, int>())
            .def("set_psf", &li::setPSF, "Sets the PSF object.")
            .def("get_psf", &li::getPSF, "Returns the PSF object.")
            .def("get_psfsq", &li::getPSFSQ)
            .def("apply_mask_flags", &li::applyMaskFlags)
            .def("apply_mask_threshold", &li::applyMaskThreshold)
            .def("sub_template", &li::subtractTemplate)
            .def("save_layers", &li::saveLayers)
            .def("save_sci", &li::saveSci)
            .def("save_mask", &li::saveMask)
            .def("save_var", &li::saveVar)
            .def("get_science", &li::getScience, "Returns the science layer raw_image.")
            .def("get_mask", &li::getMask, "Returns the mask layer raw_image.")
            .def("get_variance", &li::getVariance, "Returns the variance layer raw_image.")
            .def("set_science", &li::setScience)
            .def("set_mask", &li::setMask)
            .def("set_variance", &li::setVariance)
            .def("convolve_psf", &li::convolvePSF)
            .def("add_object", &li::addObject)
            .def("grow_mask", &li::growMask)
            .def("get_name", &li::getName, "Returns the name of the layered image.")
            .def("get_width", &li::getWidth, "Returns the image's width in pixels.")
            .def("get_height", &li::getHeight, "Returns the image's height in pixels.")
            .def("get_ppi", &li::getPPI, "Returns the image's total number of pixels.")
            .def("get_time", &li::getTime)
            .def("set_time", &li::setTime)
            .def("generate_psi_image", &li::generatePsiImage)
            .def("generate_phi_image", &li::generatePhiImage);
    py::class_<is>(m, "image_stack")
            .def(py::init<std::vector<std::string>, std::vector<pf>>())
            .def(py::init<std::vector<li>>())
            .def("get_images", &is::getImages)
            .def("get_single_image", &is::getSingleImage)
            .def("set_single_image", &is::setSingleImage)
            .def("get_times", &is::getTimes)
            .def("set_times", &is::setTimes)
            .def("img_count", &is::imgCount)
            .def("apply_mask_flags", &is::applyMaskFlags)
            .def("apply_mask_threshold", &is::applyMaskThreshold)
            .def("apply_global_mask", &is::applyGlobalMask)
            .def("grow_mask", &is::growMask)
            .def("simple_difference", &is::simpleDifference)
            .def("save_global_mask", &is::saveGlobalMask)
            .def("save_images", &is::saveImages)
            .def("get_global_mask", &is::getGlobalMask)
            .def("get_sciences", &is::getSciences)
            .def("get_masks", &is::getMasks)
            .def("get_variances", &is::getVariances)
            .def("convolve_psf", &is::convolvePSF)
            .def("get_width", &is::getWidth)
            .def("get_height", &is::getHeight)
            .def("get_ppi", &is::getPPI)
            .def("simple_shift_and_stack", &is::simpleShiftAndStack);
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
            .def("gpu_coadded_stamps",
                 (std::vector<ri>(ks::*)(std::vector<tj> &, std::vector<std::vector<bool>> &,
                                         const search::stampParameters &)) &
                         ks::coaddedScienceStampsGPU)
            // For testing
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
            .def_readwrite("x_v", &tj::xVel)
            .def_readwrite("y_v", &tj::yVel)
            .def_readwrite("lh", &tj::lh)
            .def_readwrite("flux", &tj::flux)
            .def_readwrite("x", &tj::x)
            .def_readwrite("y", &tj::y)
            .def_readwrite("obs_count", &tj::obsCount)
            .def("__repr__",
                 [](const tj &t) {
                     return "lh: " + to_string(t.lh) + " flux: " + to_string(t.flux) +
                            " x: " + to_string(t.x) + " y: " + to_string(t.y) + " x_v: " + to_string(t.xVel) +
                            " y_v: " + to_string(t.yVel) + " obs_count: " + to_string(t.obsCount);
                 })
            .def(py::pickle(
                    [](const tj &p) {  // __getstate__
                        return py::make_tuple(p.xVel, p.yVel, p.lh, p.flux, p.x, p.y, p.obsCount);
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
    py::class_<search::imageMoments>(m, "image_moments")
            .def(py::init<>())
            .def_readwrite("m00", &search::imageMoments::m00)
            .def_readwrite("m01", &search::imageMoments::m01)
            .def_readwrite("m10", &search::imageMoments::m10)
            .def_readwrite("m11", &search::imageMoments::m11)
            .def_readwrite("m02", &search::imageMoments::m02)
            .def_readwrite("m20", &search::imageMoments::m20);
    py::class_<search::stampParameters>(m, "stamp_parameters")
            .def(py::init<>())
            .def_readwrite("radius", &search::stampParameters::radius)
            .def_readwrite("stamp_type", &search::stampParameters::stamp_type)
            .def_readwrite("do_filtering", &search::stampParameters::do_filtering)
            .def_readwrite("center_thresh", &search::stampParameters::center_thresh)
            .def_readwrite("peak_offset_x", &search::stampParameters::peak_offset_x)
            .def_readwrite("peak_offset_y", &search::stampParameters::peak_offset_y)
            .def_readwrite("m01", &search::stampParameters::m01_limit)
            .def_readwrite("m10", &search::stampParameters::m10_limit)
            .def_readwrite("m11", &search::stampParameters::m11_limit)
            .def_readwrite("m02", &search::stampParameters::m02_limit)
            .def_readwrite("m20", &search::stampParameters::m20_limit);
    py::class_<bc>(m, "baryCorrection")
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

    // Functions from TrajectoryUtils (for testing)
    m.def("compute_traj_pos", &search::computeTrajPos);
    m.def("compute_traj_pos_bc", &search::computeTrajPosBC);
    m.def("ave_trajectory_dist", &search::aveTrajectoryDistance);
}
