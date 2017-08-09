#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../src/PointSpreadFunc.cpp"
#include "../src/RawImage.cpp"
#include "../src/LayeredImage.cpp"
#include "../src/ImageStack.cpp"
#include "../src/KBMOSearch.cpp"

namespace py = pybind11;

using pf = kbmod::PointSpreadFunc;
using ri = kbmod::RawImage;
using li = kbmod::LayeredImage;
using is = kbmod::ImageStack;
using ks = kbmod::KBMOSearch;
using tj = kbmod::trajectory;

using std::to_string;

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
	
	py::class_<ri>(m, "raw_image", py::buffer_protocol())
		.def_buffer([](ri &m) -> py::buffer_info {
			return py::buffer_info(
				m.getDataRef(),
				sizeof(float),
				py::format_descriptor<float>::format(),
				2,
				{ m.getHeight(), m.getWidth() },
				{ sizeof(float) * m.getHeight(),
				  sizeof(float) }
			);
		});

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
		.def("get_science", &li::getScience)
		.def("get_mask", &li::getMask)
		.def("get_variance", &li::getVariance)
		.def("convolve", &li::convolve)
		.def("add_object", &li::addObject)
		.def("get_width", &li::getWidth)
		.def("get_height", &li::getHeight)
		.def("get_ppi", &li::getPPI)
		.def("get_time", &li::getTime);
	py::class_<is>(m, "image_stack")
		.def(py::init<std::vector<std::string>>())
		.def(py::init<std::vector<li>>())
		.def("get_images", &is::getImages)
		.def("get_times", &is::getTimes)
		.def("set_times", &is::setTimes)
		.def("img_count", &is::imgCount)
		.def("apply_mask_flags", &is::applyMaskFlags)
		.def("apply_master_mask", &is::applyMasterMask)
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
	py::class_<ks>(m, "stack_search")
		.def(py::init<is, pf>())
		.def("save_psi_phi", &ks::savePsiPhi)
		.def("gpu", &ks::gpu)
		.def("filter_min_obs", &ks::filterResults)
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
		.def_readwrite("sum_count", &tj::sumCount)
		.def("__repr__", [](const tj &t) {
			return "lh: " + to_string(t.lh) + 
                            " flux: " + to_string(t.flux) + 
			       " x: " + to_string(t.x) + 
                               " y: " + to_string(t.y) + 
			      " x_v: " + to_string(t.xVel) + 
                              " y_v: " + to_string(t.yVel) +
                              " sum_count: " + to_string(t.sumCount);
			}
		);
}

