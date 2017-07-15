#include <pybind11/pybind11.h>
#include "../src/PointSpreadFunc.cpp"

namespace py = pybind11;

using pf = kbmod::PointSpreadFunc;

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
}

