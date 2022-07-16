#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../src/PointSpreadFunc.cpp"
#include "../src/RawImage.cpp"
#include "../src/LayeredImage.cpp"
#include "../src/ImageStack.cpp"
#include "../src/KBMOSearch.cpp"
#include "../src/PooledImage.cpp"

namespace py = pybind11;

using pf = kbmod::PointSpreadFunc;
using ri = kbmod::RawImage;
using li = kbmod::LayeredImage;
using is = kbmod::ImageStack;
using ks = kbmod::KBMOSearch;
using tj = kbmod::trajectory;
using td = kbmod::trajRegion;
using pi = kbmod::PooledImage;

using std::to_string;

PYBIND11_MODULE(kbmod, m) {
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
        .def("pool", &ri::pool)
        .def("pool_min", &ri::poolMin)
        .def("pool_max", &ri::poolMax)
        .def("create_stamp", &ri::createStamp)
        .def("set_pixel", &ri::setPixel)
        .def("add_pixel", &ri::addToPixel)
        .def("mask_object", &ri::maskObject)
        .def("grow_mask", &ri::growMask)
        .def("pixel_has_data", &ri::pixelHasData)
        .def("set_all", &ri::setAllPix)
        .def("get_pixel", &ri::getPixel)
        .def("get_pixel_interp", &ri::getPixelInterp)
        .def("get_ppi", &ri::getPPI)
        .def("convolve", &ri::convolve)
        .def("save_fits", &ri::saveToFile);

    py::class_<li>(m, "layered_image")
        .def(py::init<const std::string>())
        .def(py::init<std::string, int, int, 
            double, float, float>())
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
        .def("convolve", &li::convolve)
        .def("get_science_pooled", &li::poolScience)
        .def("get_variance_pooled", &li::poolVariance)
        .def("add_object", &li::addObject)
        .def("mask_object", &li::maskObject)
        .def("grow_mask", &li::growMask)
        .def("get_name", &li::getName)
        .def("get_width", &li::getWidth)
        .def("get_height", &li::getHeight)
        .def("get_ppi", &li::getPPI)
        .def("get_time", &li::getTime);
    py::class_<is>(m, "image_stack")
        .def(py::init<std::vector<std::string>>())
        .def(py::init<std::vector<li>>())
        .def("get_images", &is::getImages)
        .def("get_single_image", &is::getSingleImage)
        .def("get_times", &is::getTimes)
        .def("set_times", &is::setTimes)
        .def("img_count", &is::imgCount)
        .def("apply_mask_flags", &is::applyMaskFlags)
        .def("apply_mask_threshold", &is::applyMaskThreshold)
        .def("apply_master_mask", &is::applyMasterMask)
        .def("grow_mask", &is::growMask)
        .def("simple_difference", &is::simpleDifference)
        .def("save_master_mask", &is::saveMasterMask)
        .def("save_images", &is::saveImages)
        .def("get_master_mask", &is::getMasterMask)
        .def("get_sciences", &is::getSciences)
        .def("get_masks", &is::getMasks)
        .def("get_variances", &is::getVariances)
        .def("convolve", &is::convolve)
        .def("get_width", &is::getWidth)
        .def("get_height", &is::getHeight)
        .def("get_ppi", &is::getPPI);
    py::class_<pi>(m, "pooled_image")
        .def(py::init<ri, int>())
        .def("num_levels", &pi::numLevels)
        .def("get_base_height", &pi::getBaseHeight)
        .def("get_base_width", &pi::getBaseWidth)
        .def("get_base_ppi", &pi::getBasePPI)
        .def("get_images", &pi::getImages)
        .def("get_image", &pi::getImage)
        .def("get_pixel", &pi::getPixel)
        .def("get_mapped_pixel_at_depth", &pi::getMappedPixelAtDepth)
        .def("repool_area", &pi::repoolArea);
    m.def("pool_multiple_images", &kbmod::PoolMultipleImages);
    py::class_<ks>(m, "stack_search")
        .def(py::init<is &, pf &>())
        .def("save_psi_phi", &ks::savePsiPhi)
        .def("gpu", &ks::gpu)
        .def("gpuFilter", &ks::gpuFilter)
        .def("region_search", &ks::regionSearch)
        .def("set_debug", &ks::setDebug)
        .def("filter_min_obs", &ks::filterResults)
        // For testing
        .def("extreme_in_region", &ks::findExtremeInRegion)
        .def("biggest_fit", &ks::biggestFit)
        .def("subdivide", &ks::subdivide)
        .def("filter_bounds", &ks::filterBounds)
        .def("square_sdf", &ks::squareSDF)
        .def("filter_lh", &ks::filterLH)
        .def("pixel_extreme", &ks::pixelExtreme)
        .def("stacked_sci", (ri (ks::*)(tj &, int)) &ks::stackedScience, "set")
        .def("stacked_sci", (ri (ks::*)(td &, int)) &ks::stackedScience, "set")
        .def("median_stamps", (std::vector<ri> (ks::*)(std::vector<tj>, std::vector<std::vector<int>>, int)) &ks::medianStamps)
        .def("summed_stamps", (std::vector<ri> (ks::*)(std::vector<tj>, int)) &ks::summedStamps)
        .def("sci_stamps", (std::vector<ri> (ks::*)(tj &, int)) &ks::scienceStamps, "set")
        .def("psi_stamps", (std::vector<ri> (ks::*)(tj &, int)) &ks::psiStamps, "set2")
        .def("phi_stamps", (std::vector<ri> (ks::*)(tj &, int)) &ks::phiStamps, "set3")
        .def("sci_stamps", (std::vector<ri> (ks::*)(td &, int)) &ks::scienceStamps, "set4")
        .def("psi_stamps", (std::vector<ri> (ks::*)(td &, int)) &ks::psiStamps, "set5")
        .def("phi_stamps", (std::vector<ri> (ks::*)(td &, int)) &ks::phiStamps, "set6")
        .def("psi_curves", (std::vector<float> (ks::*)(tj &)) &ks::psiCurves)
        .def("phi_curves", (std::vector<float> (ks::*)(tj &)) &ks::phiCurves)
        .def("get_psi_images", &ks::getPsiImages)
        .def("get_phi_images", &ks::getPhiImages)
        .def("get_psi_pooled", &ks::getPsiPooled)
        .def("get_phi_pooled", &ks::getPhiPooled)
        .def("clear_psi_phi", &ks::clearPsiPhi)
        .def("prepare_psi_phi", &ks::preparePsiPhi)
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
}

