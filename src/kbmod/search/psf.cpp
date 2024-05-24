#include "psf.h"

namespace search {
PSF::PSF() : kernel(1, 1.0) {
    dim = 1;
    radius = 0;
    width = 1e-12;
    sum = 1.0;
}

PSF::PSF(float stdev) {
    if (stdev <= 0.0) {
        throw std::runtime_error("PSF stdev must be > 0.0.");
    }

    width = stdev;
    float simple_gauss[MAX_KERNEL_RADIUS];
    double psf_coverage = 0.0;
    double norm_factor = stdev * sqrt(2.0);
    int i = 0;

    // Create 1D gaussian array
    while (psf_coverage < 0.98 && i < MAX_KERNEL_RADIUS) {
        float current_bin =
                0.5 * (std::erf((float(i) + 0.5) / norm_factor) - std::erf((float(i) - 0.5) / norm_factor));
        simple_gauss[i] = current_bin;
        if (i == 0) {
            psf_coverage += current_bin;
        } else {
            psf_coverage += 2.0 * current_bin;
        }
        i++;
    }

    radius = i - 1;
    dim = 2 * radius + 1;

    // Create 2D gaussain by multiplying with itself
    kernel = std::vector<float>();
    for (int ii = 0; ii < dim; ++ii) {
        for (int jj = 0; jj < dim; ++jj) {
            float current = simple_gauss[abs(radius - ii)] * simple_gauss[abs(radius - jj)];
            kernel.push_back(current);
        }
    }
    calc_sum();
}

// Copy constructor.
PSF::PSF(const PSF& other) {
    kernel = other.kernel;
    dim = other.dim;
    radius = other.radius;
    width = other.width;
    sum = other.sum;
}

// Copy assignment.
PSF& PSF::operator=(const PSF& other) {
    kernel = other.kernel;
    dim = other.dim;
    radius = other.radius;
    width = other.width;
    sum = other.sum;
    return *this;
}

// Move constructor.
PSF::PSF(PSF&& other)
        : kernel(std::move(other.kernel)),
          dim(other.dim),
          radius(other.radius),
          width(other.width),
          sum(other.sum) {}

// Move assignment.
PSF& PSF::operator=(PSF&& other) {
    if (this != &other) {
        kernel = std::move(other.kernel);
        dim = other.dim;
        radius = other.radius;
        width = other.width;
        sum = other.sum;
    }
    return *this;
}

void PSF::calc_sum() {
    sum = 0.0;
    for (auto& i : kernel) {
        if (std::isnan(i) || std::isinf(i)) {
            throw std::runtime_error("Invalid value in PSF kernel (NaN or inf)");
        }
        sum += i;
    }
}

void PSF::square_psf() {
    for (float& i : kernel) {
        i = i * i;
    }
    calc_sum();
}

bool PSF::is_close(const PSF& img_b, float atol) const {
    const int len = kernel.size();
    if (len != img_b.kernel.size()) return false;

    for (int i = 0; i < len; ++i) {
        if (fabs(kernel[i] - img_b.kernel[i]) > atol) return false;
    }
    return true;
}
    
std::string PSF::print() {
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(3);
    for (int row = 0; row < dim; ++row) {
        ss << "| ";
        for (int col = 0; col < dim; ++col) {
            ss << kernel[row * dim + col] << " | ";
        }
        ss << "\n ";
        for (int space = 0; space < dim * 8 - 1; ++space) ss << "-";
        ss << "\n";
    }
    ss << 100.0 * sum << "% of PSF contained within kernel\n";
    return ss.str();
}

std::string PSF::stats_string() const {
    std::stringstream result;

    if (width > 0) {
        result << "PSF (Gaussian): std=" << width << ", ";
    } else {
        result << "PSF (Manual): ";
    }
    result << "radius = " << radius << ", total probability = " << sum;

    return result.str();
}

#ifdef Py_PYTHON_H
PSF::PSF(pybind11::array_t<float> arr) { set_array(arr); }

void PSF::set_array(pybind11::array_t<float> arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2)
        throw std::runtime_error(
                "Array must have 2 dimensions. (It "
                " must also be a square with odd dimensions)");

    if (info.shape[0] != info.shape[1])
        throw std::runtime_error(
                "Array must be square (x-dimension == y-dimension)."
                "It also must have odd dimensions.");
    float* pix = static_cast<float*>(info.ptr);
    dim = info.shape[0];
    if (dim % 2 == 0)
        throw std::runtime_error(
                "Array dimension must be odd. The "
                "middle of an even numbered array is ambiguous.");
    radius = dim / 2;  // Rounds down
    sum = 0.0;
    kernel = std::vector<float>(pix, pix + dim * dim);
    calc_sum();
    width = 0.0;
}

static void psf_bindings(py::module& m) {
    using psf = search::PSF;

    py::class_<psf>(m, "PSF", py::buffer_protocol(), pydocs::DOC_PSF)
            .def_buffer([](psf& m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),                                       // void *ptr;
                        sizeof(float),                                  // py::ssize_t itemsize;
                        py::format_descriptor<float>::format(),         // std::string format;
                        2,                                              // py::ssize_t ndim;
                        {m.get_dim(), m.get_dim()},                     // std::vector<py::ssize_t> shape;
                        {sizeof(float) * m.get_dim(), sizeof(float)});  // std::vector<py::ssize_t> strides;
            })
            .def(py::init<>())
            .def(py::init<float>())
            .def(py::init<py::array_t<float>>())
            .def(py::init<psf&>())
            .def("__str__", &psf::print)
            .def("set_array", &psf::set_array, pydocs::DOC_PSF_set_array)
            .def("get_std", &psf::get_std, pydocs::DOC_PSF_get_std)
            .def("get_sum", &psf::get_sum, pydocs::DOC_PSF_get_sum)
            .def("get_dim", &psf::get_dim, pydocs::DOC_PSF_get_dim)
            .def("get_radius", &psf::get_radius, pydocs::DOC_PSF_get_radius)
            .def("get_size", &psf::get_size, pydocs::DOC_PSF_get_size)
            .def("get_kernel", &psf::get_kernel, pydocs::DOC_PSF_get_kernel)
            .def("get_value", &psf::get_value, pydocs::DOC_PSF_get_value)
            .def("square_psf", &psf::square_psf, pydocs::DOC_PSF_square_psf)
            .def("print", &psf::print, pydocs::DOC_PSF_print)
            .def("is_close", &psf::is_close, pydocs::DOC_PSF_is_close)
            .def("stats_string", &psf::stats_string, pydocs::DOC_PSF_stats_string);
}
#endif

} /* namespace search */
