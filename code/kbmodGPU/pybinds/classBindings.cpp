#include <pybind11/pybind11.h>
#include "../src/PointSpreadFunc.cpp"
#include "../src/RawImage.cpp"
#include "../src/LayeredImage.cpp"

namespace py = pybind11;

using pf = kbmod::PointSpreadFunc;
using li = kbmod::LayeredImage;

PYBIND11_MODULE(kbmod, m) {
	py::class_<pf>(m, "psf")
		.def(py::init<float>())
		.def("getStdev", &pf::getStdev)
		.def("getSum", &pf::getSum)
		.def("getDim", &pf::getDim)
		.def("getRadius", &pf::getRadius)
		.def("getSize", &pf::getSize)
		.def("squarePSF", &pf::squarePSF)
		.def("printPSF", &pf::printPSF);

	py::class_<li>(m, "layered_image")
		.def(py::init<const std::string>())
		.def("applyMaskFlags", &li::applyMaskFlags)
		.def("saveSci", &li::saveSci)
		.def("saveMask", &li::saveMask)
		.def("saveVar", &li::saveVar)
		.def("convolve", &li::convolve)
		.def("getWidth", &li::getWidth)
		.def("getHeight", &li::getHeight)
		.def("getPPI", &li::getPPI)
		.def("getTime", &li::getTime);

}

