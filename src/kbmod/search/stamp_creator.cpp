#include "stamp_creator.h"
namespace search {

StampCreator::StampCreator() {}

std::vector<RawImage> StampCreator::create_stamps(ImageStack stack, const Trajectory& trj, int radius,
                                                   bool keep_no_data, const std::vector<bool>& use_index) {
    if (use_index.size() > 0 && use_index.size() != stack.img_count()) {
      throw std::runtime_error("Wrong size use_index passed into create_stamps()");
    }
    bool use_all_stamps = use_index.size() == 0;

    std::vector<RawImage> stamps;
    int num_times = stack.img_count();
    for (int i = 0; i < num_times; ++i) {
      if (use_all_stamps || use_index[i]) {
        // Calculate the trajectory position.
        float time = stack.get_zeroed_time(i);
        PixelPos pos = trj.get_pos(time);
        RawImage& img = stack.get_single_image(i).get_science();
        stamps.push_back(img.create_stamp(pos.x, pos.y, radius, keep_no_data));
      }
    }
    return stamps;
  }

// For stamps used for visualization we interpolate the pixel values, replace
// NO_DATA tages with zeros, and return all the stamps (regardless of whether
// individual timesteps have been filtered).
std::vector<RawImage> StampCreator::get_stamps(ImageStack stack, const Trajectory& t, int radius) {
    std::vector<bool> empty_vect;
    return create_stamps(stack, t, radius, false /*=keep_no_data*/, empty_vect);
}

// For creating coadded stamps, we do not interpolate the pixel values and keep
// NO_DATA tagged (so we can filter it out of mean/median).
RawImage StampCreator::get_median_stamp(ImageStack stack, const Trajectory& trj, int radius,
                                       const std::vector<bool>& use_index) {
    return create_median_image(create_stamps(stack, trj, radius, true /*=keep_no_data*/, use_index));
}

// For creating coadded stamps, we do not interpolate the pixel values and keep
// NO_DATA tagged (so we can filter it out of mean/median).
RawImage StampCreator::get_mean_stamp(ImageStack stack, const Trajectory& trj, int radius, const std::vector<bool>& use_index) {
    return create_mean_image(create_stamps(stack, trj, radius, true /*=keep_no_data*/, use_index));
}

// For creating summed stamps, we do not interpolate the pixel values and replace NO_DATA
// with zero (which is the same as filtering it out for the sum).
RawImage StampCreator::get_summed_stamp(ImageStack stack, const Trajectory& trj, int radius,
                                       const std::vector<bool>& use_index) {
    return create_summed_image(create_stamps(stack, trj, radius, false /*=keep_no_data*/, use_index));
}

#ifdef Py_PYTHON_H
  static void stamp_creator_bindings(py::module &m) {
    using sc = search::StampCreator;

    py::class_<sc>(m, "StampCreator", pydocs::DOC_StampCreator)
      .def(py::init<>())
      .def_static("get_stamps", &sc::get_stamps, pydocs::DOC_StampCreator_get_stamps)
      .def_static("get_median_stamp", &sc::get_median_stamp, pydocs::DOC_StampCreator_get_median_stamp)
      .def_static("get_mean_stamp", &sc::get_mean_stamp, pydocs::DOC_StampCreator_get_mean_stamp)
      .def_static("get_summed_stamp", &sc::get_summed_stamp, pydocs::DOC_StampCreator_get_summed_stamp);
  }
#endif /* Py_PYTHON_H */

} /* namespace search */
