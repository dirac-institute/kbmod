#include "raw_image.h"

namespace search {
using Index = indexing::Index;
using Point = indexing::Point;

RawImage::RawImage() : width(0), height(0), image() {}

RawImage::RawImage(Image& img) {
    image = std::move(img);
    height = image.rows();
    width = image.cols();
}

RawImage::RawImage(unsigned w, unsigned h, float value) : width(w), height(h) {
    if (value != 0.0f)
        image = Image::Constant(height, width, value);
    else
        image = Image::Zero(height, width);
}

// Copy constructor
RawImage::RawImage(const RawImage& old) noexcept {
    width = old.get_width();
    height = old.get_height();
    image = old.get_image();
}

// Move constructor
RawImage::RawImage(RawImage&& source) noexcept
        : width(source.width), height(source.height), image(std::move(source.image)) {}

// Copy assignment
RawImage& RawImage::operator=(const RawImage& source) noexcept {
    width = source.width;
    height = source.height;
    image = source.image;
    return *this;
}

// Move assignment
RawImage& RawImage::operator=(RawImage&& source) noexcept {
    if (this != &source) {
        width = source.width;
        height = source.height;
        image = std::move(source.image);
    }
    return *this;
}

void RawImage::replace_masked_values(float value) {
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            if (!pixel_value_valid(image(y, x))) {
                image(y, x) = value;
            }
        }
    }
}

RawImage RawImage::create_stamp(const Point& p, const int radius, const bool keep_no_data) const {
    if (radius < 0) throw std::runtime_error("stamp radius must be at least 0");

    const int dim = radius * 2 + 1;
    Image stamp = Image::Constant(dim, dim, NO_DATA);

    // Eigen gets unhappy if the stamp does not overlap at all. In this case, skip
    // the computation and leave the entire stamp set to NO_DATA. We use (int) casting here
    // instead of p.to_index() because of how we are defining the pixel grid and the center pixel.
    int idx_j = (int)p.x;
    int idx_i = (int)p.y;
    if ((idx_j + radius >= 0) && (idx_j - radius < (int)width) && (idx_i + radius >= 0) &&
        (idx_i - radius < (int)height)) {
        auto [corner, anchor, w, h] = indexing::anchored_block({idx_i, idx_j}, radius, width, height);
        stamp.block(anchor.i, anchor.j, h, w) = image.block(corner.i, corner.j, h, w);
    }

    RawImage result = RawImage(stamp);
    if (!keep_no_data) result.replace_masked_values(0.0);
    return result;
}

void RawImage::convolve(Image& psf) {
    Image result = convolve_image(image, psf);
    image = std::move(result);
}

void RawImage::apply_mask(int flags, const RawImage& mask) {
    for (unsigned int j = 0; j < height; ++j) {
        for (unsigned int i = 0; i < width; ++i) {
            int pix_flags = static_cast<int>(mask.image(j, i));
            if ((flags & pix_flags) != 0) {
                image(j, i) = NO_DATA;
            }
        }  // for i
    }      // for j
}

void RawImage::set_all(float value) { image.setConstant(value); }


#ifdef Py_PYTHON_H
static void raw_image_bindings(py::module& m) {
    using rie = search::RawImage;

    py::class_<rie>(m, "RawImage", pydocs::DOC_RawImage)
            .def(py::init<>())
            .def(py::init<search::RawImage&>())
            .def(py::init<search::Image&>(), py::arg("img").noconvert(true))
            .def(py::init<unsigned, unsigned, float>(), py::arg("w"), py::arg("h"), py::arg("value") = 0.0f)
            // attributes and properties
            .def_property_readonly("height", &rie::get_height)
            .def_property_readonly("width", &rie::get_width)
            .def_property_readonly("npixels", &rie::get_npixels)
            .def_property("image", py::overload_cast<>(&rie::get_image, py::const_), &rie::set_image)
            .def_property("imref", py::overload_cast<>(&rie::get_image), &rie::set_image)
            // pixel accessors and setters
            .def("get_pixel", &rie::get_pixel, pydocs::DOC_RawImage_get_pixel)
            .def("pixel_has_data", &rie::pixel_has_data, pydocs::DOC_RawImage_pixel_has_data)
            .def("set_pixel", &rie::set_pixel, pydocs::DOC_RawImage_set_pixel)
            .def("mask_pixel", &rie::mask_pixel, pydocs::DOC_RawImage_mask_pixel)
            .def("set_all", &rie::set_all, pydocs::DOC_RawImage_set_all)
            .def("contains_index", py::overload_cast<const indexing::Index&>(&rie::contains, py::const_),
                 pydocs::DOC_RawImage_contains_index)
            .def("contains_point", py::overload_cast<const indexing::Point&>(&rie::contains, py::const_),
                 pydocs::DOC_RawImage_contains_point)
            // python interface adapters (avoids having to construct Index & Point)
            .def("get_pixel",
                 [](rie& cls, int i, int j) {
                     return cls.get_pixel({i, j});
                 })
            .def("pixel_has_data",
                 [](rie& cls, int i, int j) {
                     return cls.pixel_has_data({i, j});
                 })
            .def("set_pixel",
                 [](rie& cls, int i, int j, double val) {
                     cls.set_pixel({i, j}, val);
                 })
            .def("mask_pixel",
                 [](rie& cls, int i, int j) {
                     cls.mask_pixel({i, j});
                 })
            .def("contains_index",
                 [](rie& cls, int i, int j) {
                     return cls.contains(indexing::Index({i, j}));
                 })
            .def("contains_point",
                 [](rie& cls, float x, float y) {
                     return cls.contains(indexing::Point({x, y}));
                 })
            // methods
            .def("replace_masked_values", &rie::replace_masked_values, py::arg("value") = 0.0f,
                 pydocs::DOC_RawImage_replace_masked_values)
            .def("create_stamp", &rie::create_stamp, pydocs::DOC_RawImage_create_stamp)
            .def("apply_mask", &rie::apply_mask, pydocs::DOC_RawImage_apply_mask)
            .def("convolve", &rie::convolve, pydocs::DOC_RawImage_convolve)
            // python interface adapters
            .def("create_stamp", [](rie& cls, float x, float y, int radius, bool keep_no_data) {
                return cls.create_stamp({x, y}, radius, keep_no_data);
            });
}
#endif

} /* namespace search */
