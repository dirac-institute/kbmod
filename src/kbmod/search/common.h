#ifndef COMMON_H_
#define COMMON_H_


#include <string>


namespace py = pybind11;


namespace search {
#ifdef HAVE_CUDA
  constexpr bool HAVE_GPU = true;
#else
  constexpr bool HAVE_GPU = false;
#endif

  constexpr unsigned int MAX_KERNEL_RADIUS = 15;
  constexpr unsigned short MAX_STAMP_EDGE = 64;
  constexpr unsigned short CONV_THREAD_DIM = 32;
  constexpr unsigned short THREAD_DIM_X = 128;
  constexpr unsigned short THREAD_DIM_Y = 2;
  constexpr unsigned short RESULTS_PER_PIXEL = 8;
  constexpr float NO_DATA = -9999.0;

  enum StampType { STAMP_SUM = 0, STAMP_MEAN, STAMP_MEDIAN };

  /*
   * Data structure to represent an objects trajectory
   * through a stack of images
   */
  struct Trajectory {
    // Trajectory velocities
    float vx;
    float vy;
    // Likelihood
    float lh;
    // Est. Flux
    float flux;
    // Origin
    short x;
    short y;
    // Number of images summed
    short obs_count;

    const std::string to_string() const {
      return "lh: " + std::to_string(lh) +
        " flux: " + std::to_string(flux) +
        " x: " + std::to_string(x) +
        " y: " + std::to_string(y) +
        " vx: " + std::to_string(vx) +
        " vy: " + std::to_string(vy) +
        " obs_count: " + std::to_string(obs_count);
    }

    // returns a yaml-compliant string
    const std::string to_yaml() const {
      return "{lh: " + std::to_string(lh) +
        ", flux: " + std::to_string(flux) +
        ", x: " + std::to_string(x) +
        ", y: " + std::to_string(y) +
        ", vx: " + std::to_string(vx) +
        ", vy: " + std::to_string(vy) +
        ", obs_count: " + std::to_string(obs_count)
        +"}";
    }
  };

  // The position (in pixels) of a trajectory.
  struct PixelPos {
    float x;
    float y;
  };

  /*
   * Linear approximation to the barycentric correction needed to transform a
   * pixel in the first image to a pixel in a consequent image. One struct needed
   * per image. Correction calculated in higher level code.
   */
  struct BaryCorrection {
    // linear coefficients of linear fit of pixel dependence
    float dx;
    float dxdx;
    float dxdy;
    float dy;
    float dydx;
    float dydy;
  };

  /* The parameters to use for the on device search. */

  struct SearchParameters {
    // Basic filtering paramets.
    int min_observations;
    float min_lh;

    // Parameters for sigmaG filtering on device.
    bool do_sigmag_filter;
    float sgl_L;
    float sgl_H;
    float sigmag_coeff;

    // Do barycentric corrections.
    bool use_corr;

    // Use a compressed image representation.
    int psi_num_bytes;  // -1 (No encoding), 1 or 2
    int phi_num_bytes;  // -1 (No encoding), 1 or 2

    // The bounds on which x and y pixels can be used
    // to start a search.
    int x_start_min;
    int x_start_max;
    int y_start_min;
    int y_start_max;

    // Provide debugging output.
    bool debug;
  };

  struct scaleParameters {
    float min_val;
    float max_val;
    float scale;
  };

  // Search data on a per-image basis.
  struct PerImageData {
    int num_images = 0;

    float* image_times = nullptr;
    BaryCorrection* bary_corrs = nullptr;

    scaleParameters* psi_params = nullptr;
    scaleParameters* phi_params = nullptr;
  };

  struct StampParameters {
    int radius = 10;
    StampType stamp_type = STAMP_SUM;
    bool do_filtering = false;

    // Thresholds on the location of the image peak.
    float center_thresh;
    float peak_offset_x;
    float peak_offset_y;

    // Limits on the moments.
    float m01_limit;
    float m10_limit;
    float m11_limit;
    float m02_limit;
    float m20_limit;
  };

  // Basic image moments use for analysis.
  struct ImageMoments {
    float m00;
    float m01;
    float m10;
    float m11;
    float m02;
    float m20;
  };

#ifdef Py_PYTHON_H

  static void trajectory_bindings(py::module &m) {
    using tj = Trajectory;

    py::class_<tj>(m, "Trajectory", R"pbdoc(
            A trajectory structure holding basic information about potential results.
            )pbdoc")
      .def(py::init<>())
      .def_readwrite("vx", &tj::vx)
      .def_readwrite("vy", &tj::vy)
      .def_readwrite("lh", &tj::lh)
      .def_readwrite("flux", &tj::flux)
      .def_readwrite("x", &tj::x)
      .def_readwrite("y", &tj::y)
      .def_readwrite("obs_count", &tj::obs_count)
      .def("__repr__", [](const tj &t) { return "Trajectory" + t.to_string(); })
      .def("__str__", &tj::to_string)
      .def(py::pickle(
                      [](const tj &p) {  // __getstate__
                        return py::make_tuple(p.vx, p.vy, p.lh, p.flux,
                                              p.x, p.y, p.obs_count);
                      },
                      [](py::tuple t) {  // __setstate__
                        if (t.size() != 7) throw std::runtime_error("Invalid state!");
                        tj trj = {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(),
                          t[3].cast<float>(), t[4].cast<short>(), t[5].cast<short>(),
                          t[6].cast<short>()};
                        return trj;
                      }));
  }

#endif /* Py_PYTHON_H */

} /* namespace search */

#endif /* COMMON_H_ */
