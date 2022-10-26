#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../src/PointSpreadFunc.cpp"
#include "../src/RawImage.cpp"
#include "../src/LayeredImage.cpp"
#include "../src/ImageStack.cpp"
#include "../src/KBMOSearch.cpp"
#include "../src/PooledImage.cpp"
#include "../src/Filtering.cpp"
#include "../src/TrajectoryUtils.cpp"

namespace py = pybind11;

using pf = kbmod::PointSpreadFunc;
using ri = kbmod::RawImage;
using li = kbmod::LayeredImage;
using is = kbmod::ImageStack;
using ks = kbmod::KBMOSearch;
using tj = kbmod::trajectory;
using bc = kbmod::baryCorrection;
using td = kbmod::trajRegion;
using pp = kbmod::pixelPos;
using pi = kbmod::PooledImage;

using std::to_string;

PYBIND11_MODULE(kbmod, m) {
    m.attr("KB_NO_DATA") = pybind11::float_(kbmod::NO_DATA);
    py::class_<pf>(m, "psf", py::buffer_protocol())
        .def_buffer([](pf &m) -> py::buffer_info {
            return py::buffer_info(
                m.kernelData(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                { m.getDim(), m.getDim() },
                { sizeof(float) * m.getDim(),
                  sizeof(float) }
            );
        })
        .def(py::init<float>())
        .def(py::init<py::array_t<float>>())
        .def(py::init<pf &>())
        .def("set_array", &pf::setArray)
        .def("get_stdev", &pf::getStdev)
        .def("get_sum", &pf::getSum)
        .def("get_dim", &pf::getDim)
        .def("get_radius", &pf::getRadius)
        .def("get_size", &pf::getSize)
        .def("get_kernel", &pf::getKernel)
        .def("square_psf", &pf::squarePSF)
        .def("print_psf", &pf::printPSF);
    
    py::class_<ri>(m, "raw_image", py::buffer_protocol())
        .def_buffer([](ri &m) -> py::buffer_info {
            return py::buffer_info(
                m.getDataRef(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                { m.getHeight(), m.getWidth() },
                { sizeof(float) * m.getWidth(),
                  sizeof(float) }
            );
        })
        .def(py::init<int, int>())
        .def(py::init<py::array_t<float>>())
        .def("get_height", &ri::getHeight)
        .def("get_width", &ri::getWidth)
        .def("get_ppi", &ri::getPPI)
        .def("set_array", &ri::setArray)
        .def("compute_bounds", &ri::computeBounds)
        .def("pool", &ri::pool)
        .def("pool_min", &ri::poolMin)
        .def("pool_max", &ri::poolMax)
        .def("create_stamp", &ri::createStamp)
        .def("set_pixel", &ri::setPixel)
        .def("add_pixel", &ri::addToPixel)
        .def("apply_mask", &ri::applyMask)
        .def("mask_object", &ri::maskObject)
        .def("grow_mask", &ri::growMask)
        .def("pixel_has_data", &ri::pixelHasData)
        .def("set_all", &ri::setAllPix)
        .def("get_pixel", &ri::getPixel)
        .def("get_pixel_interp", &ri::getPixelInterp)
        .def("get_ppi", &ri::getPPI)
        .def("extreme_in_region", &ri::extremeInRegion)
        .def("convolve", &ri::convolve)
        .def("save_fits", &ri::saveToFile);
    m.def("create_median_image", &kbmod::createMedianImage);
    m.def("create_summed_image", &kbmod::createSummedImage);
    m.def("create_mean_image", &kbmod::createMeanImage);
    py::class_<li>(m, "layered_image")
        .def(py::init<const std::string, pf&>())
        .def(py::init<std::string, int, int, 
            double, float, float, pf&>())
        .def("set_psf", &li::setPSF)
        .def("get_psf", &li::getPSF)
        .def("get_psfsq", &li::getPSFSQ)
        .def("apply_mask_flags", &li::applyMaskFlags)
        .def("apply_mask_threshold", &li::applyMaskThreshold)
        .def("sub_template", &li::subtractTemplate)
        .def("save_layers", &li::saveLayers)
        .def("save_sci", &li::saveSci)
        .def("save_mask", &li::saveMask)
        .def("save_var", &li::saveVar)
        .def("get_science", &li::getScience)
        .def("get_mask", &li::getMask)
        .def("get_variance", &li::getVariance)
        .def("set_science", &li::setScience)
        .def("set_mask", &li::setMask)
        .def("set_variance", &li::setVariance)
        .def("convolve_psf", &li::convolvePSF)
        .def("add_object", &li::addObject)
        .def("mask_object", &li::maskObject)
        .def("grow_mask", &li::growMask)
        .def("get_name", &li::getName)
        .def("get_width", &li::getWidth)
        .def("get_height", &li::getHeight)
        .def("get_ppi", &li::getPPI)
        .def("get_time", &li::getTime)
        .def("set_time", &li::setTime)
        .def("generate_psi_image", &li::generatePsiImage)
        .def("generate_phi_image", &li::generatePhiImage);
    py::class_<is>(m, "image_stack")
        .def(py::init<std::vector<std::string>, std::vector<pf> >())
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
        .def("get_ppi", &is::getPPI);
    py::class_<pi>(m, "pooled_image")
        .def(py::init<ri, int, bool>())
        .def("num_levels", &pi::numLevels)
        .def("get_base_height", &pi::getBaseHeight)
        .def("get_base_width", &pi::getBaseWidth)
        .def("get_base_ppi", &pi::getBasePPI)
        .def("get_images", &pi::getImages)
        .def("get_image", &pi::getImage)
        .def("get_pixel", &pi::getPixel)
        .def("contains_pixel", &pi::containsPixel)
        .def("get_pixel_dist_bounds", &pi::getPixelDistanceBounds)
        .def("get_mapped_pixel_at_depth", &pi::getMappedPixelAtDepth)
        .def("repool_area", &pi::repoolArea);
    m.def("pool_multiple_images", &kbmod::PoolMultipleImages);
    py::class_<ks>(m, "stack_search")
        .def(py::init<is &>())
        .def("save_psi_phi", &ks::savePsiPhi)
        .def("search", &ks::search)
        .def("enable_gpu_sigmag_filter", &ks::enableGPUSigmaGFilter)
        .def("enable_gpu_encoding", &ks::enableGPUEncoding)
        .def("enable_corr", &ks::enableCorr)
        .def("region_search", &ks::regionSearch)
        .def("set_debug", &ks::setDebug)
        .def("filter_min_obs", &ks::filterResults)
        .def("get_num_images", &ks::numImages)
        .def("get_image_stack", &ks::getImageStack)
        // For testing
        .def("get_traj_pos", &ks::getTrajPos)
        .def("get_mult_traj_pos", &ks::getMultTrajPos)
        .def("extreme_in_region", &ks::findExtremeInRegion)
        .def("filter_bounds", &ks::filterBounds)
        .def("square_sdf", &ks::squareSDF)
        .def("stacked_sci", (ri (ks::*)(tj &, int)) &ks::stackedScience, "set")
        .def("stacked_sci", (ri (ks::*)(td &, int)) &ks::stackedScience, "set")
        .def("summed_sci", (std::vector<ri> (ks::*)(std::vector<tj>, int)) &ks::summedScience)
        .def("mean_stamps", (std::vector<ri> (ks::*)(std::vector<tj>, std::vector<std::vector<int>>, int)) &ks::meanStamps)
        .def("median_stamps", (std::vector<ri> (ks::*)(std::vector<tj>, std::vector<std::vector<int>>, int)) &ks::medianStamps)
        .def("sci_stamps", (std::vector<ri> (ks::*)(tj &, int)) &ks::scienceStamps, "set")
        .def("psi_stamps", (std::vector<ri> (ks::*)(tj &, int)) &ks::psiStamps, "set2")
        .def("phi_stamps", (std::vector<ri> (ks::*)(tj &, int)) &ks::phiStamps, "set3")
        .def("sci_stamps", (std::vector<ri> (ks::*)(td &, int)) &ks::scienceStamps, "set4")
        .def("psi_stamps", (std::vector<ri> (ks::*)(td &, int)) &ks::psiStamps, "set5")
        .def("phi_stamps", (std::vector<ri> (ks::*)(td &, int)) &ks::phiStamps, "set6")
        .def("psi_curves", (std::vector<float> (ks::*)(tj &)) &ks::psiCurves)
        .def("phi_curves", (std::vector<float> (ks::*)(tj &)) &ks::phiCurves)
        .def("prepare_psi_phi", &ks::preparePsiPhi)
        .def("get_psi_images", &ks::getPsiImages)
        .def("get_phi_images", &ks::getPhiImages)
        .def("get_results", &ks::getResults)
        .def("save_results", &ks::saveResults);
    py::class_<tj>(m, "trajectory")
        .def(py::init<>())
        .def_readwrite("x_v", &tj::xVel)
        .def_readwrite("y_v", &tj::yVel)
        .def_readwrite("lh", &tj::lh)
        .def_readwrite("flux", &tj::flux)
        .def_readwrite("x", &tj::x)
        .def_readwrite("y", &tj::y)
        .def_readwrite("obs_count", &tj::obsCount)
        .def("__repr__", [](const tj &t) {
            return "lh: " + to_string(t.lh) + 
                            " flux: " + to_string(t.flux) + 
                   " x: " + to_string(t.x) + 
                               " y: " + to_string(t.y) + 
                  " x_v: " + to_string(t.xVel) + 
                              " y_v: " + to_string(t.yVel) +
                              " obs_count: " + to_string(t.obsCount);
            }
        );
    py::class_<pp>(m, "pixel_pos")
        .def(py::init<>())
        .def_readwrite("x", &pp::x)
        .def_readwrite("y", &pp::y)
        .def("__repr__", [](const pp &p) {
            return "x: " + to_string(p.x) + " y: " + to_string(p.y);
            }
        );
    py::class_<bc>(m, "baryCorrection")
        .def(py::init<>())
        .def_readwrite("dx", &bc::dx)
        .def_readwrite("dxdx", &bc::dxdx)
        .def_readwrite("dxdy", &bc::dxdy)
        .def_readwrite("dy", &bc::dy)
        .def_readwrite("dydx", &bc::dydx)
        .def_readwrite("dydy", &bc::dydy)
        .def("__repr__", [](const bc &b) {
            return "dx = " + to_string(b.dx) + " + " + to_string(b.dxdx) + " x + " + to_string(b.dxdy) + " y; "+
                " dy = " + to_string(b.dy) + " + " + to_string(b.dydx) + " x + " + to_string(b.dydy)+ " y";
            }
        );
    py::class_<td>(m, "traj_region")
        .def(py::init<>())
        .def_readwrite("ix", &td::ix)
        .def_readwrite("iy", &td::iy)
        .def_readwrite("fx", &td::fx)
        .def_readwrite("fy", &td::fy)
        .def_readwrite("depth", &td::depth)
        .def_readwrite("obs_count", &td::obs_count)
        .def_readwrite("likelihood", &td::likelihood)
        .def_readwrite("flux", &td::flux)
        .def("__repr__", [](const td &t) {
            return "ix: " + to_string(t.ix) +
                              " iy: " + to_string(t.iy) +
                  " fx: " + to_string(t.fx) +
                              " fy: " + to_string(t.fy) +
               " depth: " + to_string(static_cast<int>(t.depth)) +
                       " obs_count: " + to_string(static_cast<int>(t.obs_count)) +
                  " lh: " + to_string(t.likelihood) +
                 " flux " + to_string(t.flux);
            }
        );
    // Functions from Filtering.cpp
    m.def("sigmag_filtered_indices", &kbmod::sigmaGFilteredIndices);
    m.def("kalman_filtered_indices", &kbmod::kalmanFiteredIndices);
    m.def("calculate_likelihood_psi_phi", &kbmod::calculateLikelihoodFromPsiPhi);
    
    // Functions from TrajectoryUtils (for testing)
    m.def("compute_traj_pos", &kbmod::computeTrajPos);
    m.def("compute_traj_pos_bc", &kbmod::computeTrajPosBC);
    m.def("ave_trajectory_dist", &kbmod::aveTrajectoryDistance);
    m.def("convert_traj_region", &kbmod::convertTrajRegion);
    m.def("subdivide_traj_region", &kbmod::subdivideTrajRegion);
    m.def("filter_traj_regions_lh", &kbmod::filterTrajRegionsLH);
}

