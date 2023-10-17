#include "stamp_creator.h"


namespace search {

  StampCreator::StampCreator() {}


  std::vector<RawImage> StampCreator::create_stamps(ImageStack stack, const Trajectory& trj, int radius, bool interpolate,
                                                   bool keep_no_data, const std::vector<bool>& use_index) {
    if (use_index.size() > 0 && use_index.size() != stack.img_count()) {
      throw std::runtime_error("Wrong size use_index passed into create_stamps()");
    }
    bool use_all_stamps = use_index.size() == 0;

    std::vector<RawImage> stamps;
    int num_times = stack.img_count();
    for (int i = 0; i < num_times; ++i) {
      if (use_all_stamps || use_index[i]) {
        float time = stack.get_zeroed_time(i);
        PixelPos pos = {trj.x + time * trj.vx, trj.y + time * trj.vy};
        //PixelPos pos = get_trajectory_position(trj, i);
        RawImage& img = stack.get_single_image(i).get_science();
        stamps.push_back(img.create_stamp(pos.x, pos.y, radius, interpolate, keep_no_data));
      }
    }
    return stamps;
  }


  std::vector<RawImage> StampCreator::create_stamps(ImageStack stack, Trajectory t, int radius, const std::vector<RawImage*>& imgs,
                                                   bool interpolate) {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");
    std::vector<RawImage> stamps;
    for (int i = 0; i < imgs.size(); ++i) {
      float time = stack.get_zeroed_time(i);
      PixelPos pos = {t.x + time * t.vx, t.y + time * t.vy};
      //PixelPos pos = get_trajectory_position(t, i);
      stamps.push_back(imgs[i]->create_stamp(pos.x, pos.y, radius, interpolate, false));
    }
    return stamps;
  }

#ifdef Py_PYTHON_H
  static void stamp_creator_bindings(py::module &m) {
    using tj = search::Trajectory;
    using pf = search::PSF;
    using ri = search::RawImage;
    using is = search::ImageStack;
    using sc = search::StampCreator;
    using ks = search::StackSearch;

    py::class_<sc>(m, "StampCreator", pydocs::DOC_StampCreator)
      .def(py::init<>());
  }
#endif /* Py_PYTHON_H */

} /* namespace search */
