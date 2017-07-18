#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../src/PointSpreadFunc.cpp"
#include "../src/RawImage.cpp"
#include "../src/LayeredImage.cpp"
#include "../src/ImageStack.cpp"
#include "../src/KBMOSearch.cpp"

namespace py = pybind11;

using pf = kbmod::PointSpreadFunc;
using li = kbmod::LayeredImage;
using is = kbmod::ImageStack;
using ks = kbmod::KBMOSearch;

PYBIND11_MODULE(kbmod, m) {
	py::class_<pf>(m, "psf")
		.def(py::init<float>())
		.def("get_stdev", &pf::getStdev)
		.def("get_sum", &pf::getSum)
		.def("get_dim", &pf::getDim)
		.def("get_radius", &pf::getRadius)
		.def("get_size", &pf::getSize)
		.def("square_psf", &pf::squarePSF)
		.def("print_psf", &pf::printPSF);

	py::class_<li>(m, "layered_image")
		.def(py::init<const std::string>())
		.def(py::init<std::string, int, int, 
			double, float, float>())
		.def("apply_mask_flags", &li::applyMaskFlags)
		//.def("sci_numpy", &li::sciToNumpy)
		.def("save_layers", &li::saveLayers)
		.def("save_sci", &li::saveSci)
		.def("save_mask", &li::saveMask)
		.def("save_var", &li::saveVar)
		.def("convolve", &li::convolve)
		.def("get_width", &li::getWidth)
		.def("get_height", &li::getHeight)
		.def("get_ppi", &li::getPPI)
		.def("get_time", &li::getTime);
	py::class_<is>(m, "image_stack")
		.def(py::init<std::list<std::string>>())
		.def("get_images", &is::getImages)
		.def("img_count", &is::imgCount)
		.def("apply_mask_flags", &is::applyMaskFlags)
		.def("apply_master_mask", &is::applyMasterMask)
		.def("save_sci", &is::saveSci)
		.def("save_mask", &is::saveMask)
		.def("save_var", &is::saveVar)
		.def("convolve", &is::convolve)
		.def("get_width", &is::getWidth)
		.def("get_height", &is::getHeight)
		.def("get_ppi", &is::getPPI);
	py::class_<ks>(m, "stack_search")
		.def(py::init<kbmod::ImageStack, kbmod::PointSpreadFunc>())
		.def("image_save_location", &ks::imageSaveLocation)
		.def("gpu", &ks::gpu)
		.def("save_results", &ks::saveResults);
		
}

