#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "KBMOSearch.cpp"
#include "Filtering.cpp"

namespace py = pybind11;

using tj = trajectory;
using is = image_base::ImageStack;
using ri = image_base::RawImage;
using ks = search::KBMOSearch;

using std::to_string;

PYBIND11_MODULE(search, m) {
    py::class_<ks>(m, "stack_search")
            .def(py::init<is &>())
            .def("save_psi_phi", &ks::savePsiPhi)
            .def("search", &ks::search)
            .def("enable_gpu_sigmag_filter", &ks::enableGPUSigmaGFilter)
            .def("enable_gpu_encoding", &ks::enableGPUEncoding)
            .def("enable_corr", &ks::enableCorr)
            .def("set_start_bounds_x", &ks::setStartBoundsX)
            .def("set_start_bounds_y", &ks::setStartBoundsY)
            .def("set_debug", &ks::setDebug)
            .def("filter_min_obs", &ks::filterResults)
            .def("get_num_images", &ks::numImages)
            .def("get_image_stack", &ks::getImageStack)
            // Science Stamp Functions
            .def("science_viz_stamps", &ks::scienceStampsForViz)
            .def("median_sci_stamp", &ks::medianScienceStamp)
            .def("mean_sci_stamp", &ks::meanScienceStamp)
            .def("summed_sci_stamp", &ks::summedScienceStamp)
            .def("coadded_stamps",
                 (std::vector<ri>(ks::*)(std::vector<tj> &, std::vector<std::vector<bool>> &,
                                         const stampParameters &, bool)) &
                         ks::coaddedScienceStamps)
            // For testing
            .def("filter_stamp", &ks::filterStamp)
            .def("get_traj_pos", &ks::getTrajPos)
            .def("get_mult_traj_pos", &ks::getMultTrajPos)
            .def("psi_curves", (std::vector<float>(ks::*)(tj &)) & ks::psiCurves)
            .def("phi_curves", (std::vector<float>(ks::*)(tj &)) & ks::phiCurves)
            .def("prepare_psi_phi", &ks::preparePsiPhi)
            .def("get_psi_images", &ks::getPsiImages)
            .def("get_phi_images", &ks::getPhiImages)
            .def("get_results", &ks::getResults)
            .def("set_results", &ks::setResults);

    // Functions from Filtering.cpp
    m.def("sigmag_filtered_indices", &search::sigmaGFilteredIndices);
    m.def("calculate_likelihood_psi_phi", &search::calculateLikelihoodFromPsiPhi);
}
