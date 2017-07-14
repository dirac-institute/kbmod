#include <pybind11/pybind11.h>
#include "../src/PointSpreadFunc.cpp"

namespace py = pybind11;

int square(int x) {
	return x*x;
}

PYBIND11_PLUGIN(test) {
	py::module m("test", "this is a test");
	m.def("square", &square);
	return m.ptr();
}

