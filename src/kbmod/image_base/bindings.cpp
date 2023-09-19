#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "PointSpreadFunc.cpp"
#include "RawImage.cpp"
#include "LayeredImage.cpp"
#include "ImageStack.cpp"

namespace py = pybind11;

using pf = image_base::PointSpreadFunc;
using ri = image_base::RawImage;
using li = image_base::LayeredImage;
using is = image_base::ImageStack;

using std::to_string;

PYBIND11_MODULE(search, m) {
    m.attr("KB_NO_DATA") = pybind11::float_(NO_DATA);
    m.attr("HAS_GPU") = pybind11::bool_(HAVE_GPU);
    py::enum_<StampType>(m, "StampType")
            .value("STAMP_SUM", StampType::STAMP_SUM)
            .value("STAMP_MEAN", StampType::STAMP_MEAN)
            .value("STAMP_MEDIAN", StampType::STAMP_MEDIAN)
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
            .def("get_value", &pf::getValue, "Returns the PSF kernel value at a specific point.")
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
            .def(py::init<const ri &>())
            .def(py::init<py::array_t<float>>())
            .def("get_height", &ri::getHeight, "Returns the image's height in pixels.")
            .def("get_width", &ri::getWidth, "Returns the image's width in pixels.")
            .def("get_npixels", &ri::getNPixels, "Returns the image's total number of pixels.")
            .def("get_all_pixels", &ri::getPixels, "Returns a list of the images pixels.")
            .def("set_array", &ri::setArray, "Sets all image pixels given an array of values.")
            .def("get_obstime", &ri::getObstime, "Get the observation time of the image.")
            .def("set_obstime", &ri::setObstime, "Set the observation time of the image.")
            .def("approx_equal", &ri::approxEqual, "Checks if two images are approximately equal.")
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
            .def("convolve_cpu", &ri::convolve_cpu, "Convolve the image with a PSF.")
            .def("load_fits", &ri::loadFromFile, "Load the image data from a FITS file.")
            .def("save_fits", &ri::saveToFile, "Save the image to a FITS file.")
            .def("append_fits_layer", &ri::appendLayerToFile, "Append the image as a layer in a FITS file.");
    m.def("create_median_image", &image_base::createMedianImage);
    m.def("create_summed_image", &image_base::createSummedImage);
    m.def("create_mean_image", &image_base::createMeanImage);
    py::class_<li>(m, "layered_image")
            .def(py::init<const std::string, pf &>())
            .def(py::init<const ri &, const ri &, const ri &, pf &>(), R"pbdoc(
            Creates a layered_image out of individual `raw_image` layers.

            Parameters
            ----------
            sci : `raw_image`
                The `raw_image` for the science layer.
            var : `raw_image`
                The `raw_image` for the cariance layer.
            msk : `raw_image`
                The `raw_image` for the mask layer.
            p : `psf`
                The PSF for the image.

            Raises
            ------
            Raises an exception if the layers are not the same size.
            )pbdoc")
            .def(py::init<std::string, int, int, double, float, float, pf &>())
            .def(py::init<std::string, int, int, double, float, float, pf &, int>())
            .def("set_psf", &li::setPSF, "Sets the PSF object.")
            .def("get_psf", &li::getPSF, "Returns the PSF object.")
            .def("get_psfsq", &li::getPSFSQ)
            .def("apply_mask_flags", &li::applyMaskFlags)
            .def("apply_mask_threshold", &li::applyMaskThreshold)
            .def("sub_template", &li::subtractTemplate)
            .def("save_layers", &li::saveLayers)
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
            .def("get_npixels", &li::getNPixels, "Returns the image's total number of pixels.")
            .def("get_obstime", &li::getObstime, "Get the image's observation time.")
            .def("set_obstime", &li::setObstime, "Set the image's observation time.")
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
            .def("save_global_mask", &is::saveGlobalMask)
            .def("save_images", &is::saveImages)
            .def("get_global_mask", &is::getGlobalMask)
            .def("convolve_psf", &is::convolvePSF)
            .def("get_width", &is::getWidth)
            .def("get_height", &is::getHeight)
            .def("get_npixels", &is::getNPixels);
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
    py::class_<pixelPos>(m, "pixel_pos")
            .def(py::init<>())
            .def_readwrite("x", &pixelPos::x)
            .def_readwrite("y", &pixelPos::y)
            .def("__repr__", [](const pixelPos &p) { return "x: " + to_string(p.x) + " y: " + to_string(p.y); });
    py::class_<imageMoments>(m, "image_moments")
            .def(py::init<>())
            .def_readwrite("m00", &imageMoments::m00)
            .def_readwrite("m01", &imageMoments::m01)
            .def_readwrite("m10", &imageMoments::m10)
            .def_readwrite("m11", &imageMoments::m11)
            .def_readwrite("m02", &imageMoments::m02)
            .def_readwrite("m20", &imageMoments::m20);
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
}
