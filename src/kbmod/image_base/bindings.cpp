#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "PointSpreadFunc.cpp"
#include "psf_bindings.cpp"
#include "RawImage.cpp"
#include "LayeredImage.cpp"
#include "ImageStack.cpp"

namespace py = pybind11;

// this has to be here because all the other bindings
// f.e. for LayeredImage(PSF) - but in practice this goes
// away into individual bindings alongisde with the import
// at the top.
using pf = image_base::PointSpreadFunc;
using ri = image_base::RawImage;
using li = image_base::LayeredImage;
using is = image_base::ImageStack;

using std::to_string;

PYBIND11_MODULE(image_base, m) {
  m.attr("KB_NO_DATA") = pybind11::float_(NO_DATA);
  m.attr("HAS_GPU") = pybind11::bool_(HAVE_GPU);

    image_base_bindings::psf_bindings_factory(m);

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
}
