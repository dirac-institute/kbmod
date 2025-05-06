#include "image_stack.h"

#include <algorithm>
#include "logging.h"

namespace search {

ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) : height(0), width(0) {
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Constructing ImageStack with " + std::to_string(imgs.size()) + " images.");
    images = imgs;

    // Check that the images are all the same size.
    if (images.size() > 0) {
        width = images[0].get_width();
        height = images[0].get_height();
        for (auto& img : images) {
            assert_sizes_equal(img.get_width(), width, "ImageStack image width");
            assert_sizes_equal(img.get_height(), height, "ImageStack image height");
        }
    }
}

ImageStack::ImageStack() : height(0), width(0) {
    logging::getLogger("kbmod.search.image_stack")->debug("Constructing an empty ImageStack.");
}

ImageStack::~ImageStack() {}

LayeredImage& ImageStack::get_single_image(int index) {
    if (index < 0 || index >= images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index];
}

void ImageStack::set_single_image(int index, LayeredImage& img, bool force_move) {
    if (index < 0 || index >= images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    assert_sizes_equal(img.get_width(), width, "ImageStack image width");
    assert_sizes_equal(img.get_height(), height, "ImageStack image height");

    if (force_move) {
        images[index] = img;
    } else {
        images[index] = std::move(img);
    }
}

void ImageStack::append_image(LayeredImage& img, bool force_move) {
    if (images.size() == 0) {
        width = img.get_width();
        height = img.get_height();
    } else {
        assert_sizes_equal(img.get_width(), width, "ImageStack image width");
        assert_sizes_equal(img.get_height(), height, "ImageStack image height");
    }

    if (force_move) {
        images.push_back(std::move(img));
    } else {
        images.push_back(img);
    }
}

double ImageStack::get_obstime(int index) const {
    if (index < 0 || index >= images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index].get_obstime();
}

double ImageStack::get_zeroed_time(int index) const {
    if (index < 0 || index >= images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index].get_obstime() - images[0].get_obstime();
}

std::vector<double> ImageStack::build_zeroed_times() const {
    std::vector<double> zeroed_times = std::vector<double>();
    if (images.size() > 0) {
        double t0 = images[0].get_obstime();
        for (auto& i : images) {
            zeroed_times.push_back(i.get_obstime() - t0);
        }
    }
    return zeroed_times;
}

void ImageStack::sort_by_time() {
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Sorting " + std::to_string(images.size()) + " images by time.");
    std::sort(images.begin(), images.end(),
              [](const LayeredImage& a, const LayeredImage& b) { return a.get_obstime() < b.get_obstime(); });
}

#ifdef Py_PYTHON_H
static void image_stack_bindings(py::module& m) {
    using is = search::ImageStack;
    using li = search::LayeredImage;

    py::class_<is>(m, "ImageStack", pydocs::DOC_ImageStack)
            .def(py::init<>())
            .def(py::init<std::vector<li>>())
            .def("__len__", &is::img_count)
            .def_property_readonly("height", &is::get_height)
            .def_property_readonly("width", &is::get_width)
            .def_property_readonly("num_times", &is::img_count)
            .def("get_images", &is::get_images, py::return_value_policy::reference_internal,
                 pydocs::DOC_ImageStack_get_images)
            .def("get_single_image", &is::get_single_image, py::return_value_policy::reference_internal,
                 pydocs::DOC_ImageStack_get_single_image)
            .def("set_single_image", &is::set_single_image, py::arg("index"), py::arg("img"),
                 py::arg("force_move") = false, pydocs::DOC_ImageStack_set_single_image)
            .def("append_image", &is::append_image, py::arg("img"), py::arg("force_move") = false,
                 pydocs::DOC_ImageStack_append_image)
            .def("get_obstime", &is::get_obstime, pydocs::DOC_ImageStack_get_obstime)
            .def("get_zeroed_time", &is::get_zeroed_time, pydocs::DOC_ImageStack_get_zeroed_time)
            .def("build_zeroed_times", &is::build_zeroed_times, pydocs::DOC_ImageStack_build_zeroed_times)
            .def("sort_by_time", &is::sort_by_time, pydocs::DOC_ImageStack_sort_by_time)
            .def("img_count", &is::img_count, pydocs::DOC_ImageStack_img_count)
            .def("get_width", &is::get_width, pydocs::DOC_ImageStack_get_width)
            .def("get_height", &is::get_height, pydocs::DOC_ImageStack_get_height)
            .def("get_npixels", &is::get_npixels, pydocs::DOC_ImageStack_get_npixels)
            .def("get_total_pixels", &is::get_total_pixels, pydocs::DOC_ImageStack_get_total_pixels);
}

#endif /* Py_PYTHON_H */

} /* namespace search */
