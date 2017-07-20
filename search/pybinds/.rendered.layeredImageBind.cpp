#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(int x) {
	return x*x;
}

PYBIND_PLUGIN(layeredImageBind) {
	py::module m("layeredImageBind", "auto-compiled c++ extension");
	m.def("square", &square);
	return m.ptr();
}

