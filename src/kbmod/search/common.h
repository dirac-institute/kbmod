#ifndef COMMON_H_
#define COMMON_H_

#include <assert.h>
#include <math.h>
#include <stdexcept>
#include <string>

#include "pydocs/common_docs.h"

namespace search {
#ifdef HAVE_CUDA
constexpr bool HAVE_CUDA_LIB = true;
#else
constexpr bool HAVE_CUDA_LIB = false;
#endif

#ifdef HAVE_OPENMP
constexpr bool HAVE_OMP = true;
#else
constexpr bool HAVE_OMP = false;
#endif

constexpr unsigned int MAX_KERNEL_RADIUS = 15;
constexpr unsigned short MAX_STAMP_EDGE = 64;
constexpr unsigned short CONV_THREAD_DIM = 32;
constexpr unsigned short THREAD_DIM_X = 128;
constexpr unsigned short THREAD_DIM_Y = 2;

// Limits for the GPU specific functions.
constexpr unsigned int MAX_NUM_IMAGES = 200;
constexpr unsigned int MAX_STAMP_IMAGES = 200;

// The NO_DATA flag indicates masked values in the image.
constexpr float NO_DATA = NAN;

enum StampType { STAMP_SUM = 0, STAMP_MEAN, STAMP_MEDIAN, STAMP_VAR_WEIGHTED };

// A helper function to check that a pixel value is valid. This should include
// both masked pixel values (NO_DATA above) and other invalid values (e.g. inf).
inline bool pixel_value_valid(float value) { return std::isfinite(value); }

// A common error check with human readable output.
inline void assert_sizes_equal(size_t actual, size_t expected, std::string name) {
    if (actual != expected) {
        throw std::runtime_error("Size mismatch error [" + name + "]. Expected " + std::to_string(expected) +
                                 ". Found " + std::to_string(actual));
    }
}

/*
 * Data structure to represent an objects trajectory
 * through a stack of images
 */
struct Trajectory {
    // Trajectory velocities
    float vx = 0.0;
    float vy = 0.0;
    // Likelihood
    float lh = 0.0;
    // Est. Flux
    float flux = 0.0;
    // Origin
    int x = 0;
    int y = 0;
    // Number of images summed
    int obs_count;

    // Get pixel positions from a zero-shifted time. Centered indicates whether
    // the prediction starts from the center of the pixel (which it does in the search)
    inline float get_x_pos(double time, bool centered = true) const {
        return centered ? (x + time * vx + 0.5f) : (x + time * vx);
    }
    inline float get_y_pos(double time, bool centered = true) const {
        return centered ? (y + time * vy + 0.5f) : (y + time * vy);
    }

    inline int get_x_index(double time) const { return (int)floor(get_x_pos(time, true)); }
    inline int get_y_index(double time) const { return (int)floor(get_y_pos(time, true)); }

    const std::string to_string() const {
        return "lh: " + std::to_string(lh) + " flux: " + std::to_string(flux) + " x: " + std::to_string(x) +
               " y: " + std::to_string(y) + " vx: " + std::to_string(vx) + " vy: " + std::to_string(vy) +
               " obs_count: " + std::to_string(obs_count);
    }

    // This is a hack to provide a constructor with non-default arguments in Python. If we include
    // the constructor as a method in the Trajectory struct CUDA will complain when creating new objects
    // because it cannot call out to a host function.
    static Trajectory make_trajectory(int x, int y, float vx, float vy, float flux, float lh, int obs_count) {
        Trajectory trj;
        trj.x = x;
        trj.y = y;
        trj.vx = vx;
        trj.vy = vy;
        trj.flux = flux;
        trj.lh = lh;
        trj.obs_count = obs_count;
        return trj;
    }
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

    // Use a compressed image representation.
    int encode_num_bytes = -1;  // -1 (No encoding), 1 or 2

    // The bounds on which x and y pixels can be used
    // to start a search.
    int x_start_min;
    int x_start_max;
    int y_start_min;
    int y_start_max;

    // The number of results per pixel to return
    int results_per_pixel = 8;

    const std::string to_string() const {
        std::string output = ("Filtering Settings:\n  min_observations: " + std::to_string(min_observations) +
                              "\n  min_lh: " + std::to_string(min_lh));
        if (do_sigmag_filter) {
            output += ("\n  SigmaG: [" + std::to_string(sgl_L) + ", " + std::to_string(sgl_H) +
                       "] coeff=" + std::to_string(sigmag_coeff));
        } else {
            output += "\n  SigmaG: OFF";
        }
        output += "\nResults per pixel: " + std::to_string(results_per_pixel);
        output += "\nencode_num_bytes: " + std::to_string(encode_num_bytes);
        output += ("\nBounds X=[" + std::to_string(x_start_min) + ", " + std::to_string(x_start_max) +
                   "] Y=[" + std::to_string(y_start_min) + ", " + std::to_string(y_start_max) + "]");
        return output;
    }
};

#ifdef Py_PYTHON_H
static void trajectory_bindings(py::module &m) {
    using tj = Trajectory;

    py::class_<tj>(m, "Trajectory", pydocs::DOC_Trajectory)
            .def(py::init(&tj::make_trajectory), py::arg("x") = 0, py::arg("y") = 0, py::arg("vx") = 0.0f,
                 py::arg("vy") = 0.0f, py::arg("flux") = 0.0f, py::arg("lh") = 0.0f, py::arg("obs_count") = 0)
            .def_readwrite("vx", &tj::vx)
            .def_readwrite("vy", &tj::vy)
            .def_readwrite("lh", &tj::lh)
            .def_readwrite("flux", &tj::flux)
            .def_readwrite("x", &tj::x)
            .def_readwrite("y", &tj::y)
            .def_readwrite("obs_count", &tj::obs_count)
            .def("get_x_pos", &tj::get_x_pos, py::arg("time"), py::arg("centered") = true,
                 pydocs::DOC_Trajectory_get_x_pos)
            .def("get_y_pos", &tj::get_y_pos, py::arg("time"), py::arg("centered") = true,
                 pydocs::DOC_Trajectory_get_y_pos)
            .def("get_x_index", &tj::get_x_index, pydocs::DOC_Trajectory_get_x_index)
            .def("get_y_index", &tj::get_y_index, pydocs::DOC_Trajectory_get_y_index)
            .def("__repr__", [](const tj &t) { return "Trajectory(" + t.to_string() + ")"; })
            .def("__str__", &tj::to_string)
            .def(py::pickle(
                    [](const tj &p) {  // __getstate__
                        return py::make_tuple(p.vx, p.vy, p.lh, p.flux, p.x, p.y, p.obs_count);
                    },
                    [](py::tuple t) {  // __setstate__
                        if (t.size() != 8) throw std::runtime_error("Invalid state!");
                        tj trj = {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(),
                                  t[3].cast<float>(), t[4].cast<int>(),   t[5].cast<int>(),
                                  t[6].cast<int>()};
                        return trj;
                    }));
}

static void search_parameters_bindings(py::module &m) {
    py::class_<SearchParameters>(m, "SearchParameters")
            .def(py::init<>())
            .def("__str__", &SearchParameters::to_string)
            .def_readwrite("min_observations", &SearchParameters::min_observations)
            .def_readwrite("min_lh", &SearchParameters::min_lh)
            .def_readwrite("do_sigmag_filter", &SearchParameters::do_sigmag_filter)
            .def_readwrite("sgl_L", &SearchParameters::sgl_L)
            .def_readwrite("sgl_H", &SearchParameters::sgl_H)
            .def_readwrite("sigmag_coeff", &SearchParameters::sigmag_coeff)
            .def_readwrite("encode_num_bytes", &SearchParameters::encode_num_bytes)
            .def_readwrite("x_start_min", &SearchParameters::x_start_min)
            .def_readwrite("x_start_max", &SearchParameters::x_start_max)
            .def_readwrite("y_start_min", &SearchParameters::y_start_min)
            .def_readwrite("y_start_max", &SearchParameters::y_start_max)
            .def_readwrite("results_per_pixel", &SearchParameters::results_per_pixel);
}

#endif /* Py_PYTHON_H */

} /* namespace search */

#endif /* COMMON_H_ */
