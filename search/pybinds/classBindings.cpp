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
using td = kbmod::trajRegion;

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
		.def("set_array", &pf::setArray)
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
		})
		.def(py::init<int, int>())
		.def(py::init<py::array_t<float>>())
		.def("set_array", &ri::setArray)
		.def("pool", &ri::pool)
		.def("pool_min", &ri::poolMin)
		.def("pool_max", &ri::poolMax)
		.def("set_pixel", &ri::setPixel)
		.def("set_all", &ri::setAllPix)
		.def("get_pixel", &ri::getPixel)
		.def("get_pixel_interp", &ri::getPixelInterp)
		.def("get_ppi", &ri::getPPI)
		.def("convolve", &ri::convolve);

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
		.def("get_science_pooled", &li::poolScience)
		.def("get_variance_pooled", &li::poolVariance)
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
		.def("region_search", &ks::regionSearch)
		.def("set_debug", &ks::setDebug)
		.def("filter_min_obs", &ks::filterResults)
		// For testing
		.def("extreme_in_region", &ks::findExtremeInRegion)
		.def("biggest_fit", &ks::biggestFit)
		.def("read_pixel_depth", &ks::readPixelDepth)
		.def("subdivide", &ks::subdivide)
		.def("filter_bounds", &ks::filterBounds)
		.def("square_sdf", &ks::squareSDF)
		.def("filter_lh", &ks::filterLH)
		.def("pixel_extreme", &ks::pixelExtreme)
		.def("get_psi_images", &ks::getPsiImages)
		.def("get_phi_images", &ks::getPhiImages)
		.def("get_psi_pooled", &ks::getPsiPooled)
		.def("get_phi_pooled", &ks::getPhiPooled)
		.def("clear_psi_phi", &ks::clearPsiPhi)
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

