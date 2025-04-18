#include "image_stack.h"

#include <algorithm>
#include "logging.h"

namespace search {

ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) : data_on_gpu(false), height(0), width(0) {
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

ImageStack::ImageStack() : data_on_gpu(false), height(0), width(0) {
    logging::getLogger("kbmod.search.image_stack")->debug("Constructing an empty ImageStack.");
}

ImageStack::~ImageStack() { clear_from_gpu(); }

LayeredImage& ImageStack::get_single_image(int index) {
    if (index < 0 || index >= images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index];
}

void ImageStack::set_single_image(int index, LayeredImage& img, bool force_move) {
    if (data_on_gpu) throw std::runtime_error("Cannot modify images while on GPU");
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
    if (data_on_gpu) throw std::runtime_error("Cannot modify images while on GPU");

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
    if (data_on_gpu) throw std::runtime_error("Cannot modify images while on GPU");
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Sorting " + std::to_string(images.size()) + " images by time.");
    std::sort(images.begin(), images.end(),
              [](const LayeredImage& a, const LayeredImage& b) { return a.get_obstime() < b.get_obstime(); });
}

void ImageStack::copy_to_gpu() {
    if (data_on_gpu) return;  // Nothing to do

    logging::Logger* logger = logging::getLogger("kbmod.search.image_stack");

    // Move the time data to the GPU.
    uint64_t num_times = img_count();
    gpu_time_array.resize(num_times);
    logger->debug(stat_gpu_memory_mb());
    logger->debug("Copying times to GPU. " + gpu_time_array.stats_string());

    std::vector<double> image_times = build_zeroed_times();
    gpu_time_array.copy_vector_to_gpu(image_times);

    // Move the image data to the GPU.
    uint64_t height = get_height();
    uint64_t width = get_width();
    uint64_t img_pixels = height * width;
    gpu_image_array.resize(img_pixels * num_times);
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Copying images to GPU. " + gpu_image_array.stats_string());

    // Copy the data into a single block of GPU memory one image at a time.
    try {
        for (uint64_t t = 0; t < num_times; ++t) {
            float* img_ptr = get_single_image(t).get_science().data();
            uint64_t start_index = t * img_pixels;
            gpu_image_array.copy_array_into_subset_of_gpu(img_ptr, start_index, img_pixels);
        }
    } catch (const std::runtime_error& error) {
        // If anything fails clear all GPU memory (times and any partial image).
        clear_from_gpu();
        throw;
    }
    logger->debug(stat_gpu_memory_mb());

    // Check if we failed with either array.
    if (!gpu_image_array.on_gpu() || !gpu_image_array.on_gpu()) {
        clear_from_gpu();
        throw std::runtime_error("Failed to copy image data to GPU.");
    }

    // Mark the data as copied.
    data_on_gpu = true;
}

void ImageStack::clear_from_gpu() {
    logging::Logger* logger = logging::getLogger("kbmod.search.image_stack");
    logger->debug(stat_gpu_memory_mb());

    if (gpu_image_array.on_gpu()) {
        logger->debug("Freeing images on GPU. " + gpu_image_array.stats_string());
        gpu_image_array.free_gpu_memory();
    }

    if (gpu_time_array.on_gpu()) {
        logger->debug("Freeing times on GPU: " + gpu_time_array.stats_string());
        gpu_time_array.free_gpu_memory();
    }

    logger->debug(stat_gpu_memory_mb());
    data_on_gpu = false;
}

#ifdef Py_PYTHON_H
static void image_stack_bindings(py::module& m) {
    using is = search::ImageStack;
    using li = search::LayeredImage;

    py::class_<is>(m, "ImageStack", pydocs::DOC_ImageStack)
            .def(py::init<>())
            .def(py::init<std::vector<li>>())
            .def_property_readonly("on_gpu", &is::on_gpu, pydocs::DOC_ImageStack_on_gpu)
            .def("__len__", &is::img_count)
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
            .def("get_total_pixels", &is::get_total_pixels, pydocs::DOC_ImageStack_get_total_pixels)
            .def("copy_to_gpu", &is::copy_to_gpu, pydocs::DOC_ImageStack_copy_to_gpu)
            .def("clear_from_gpu", &is::clear_from_gpu, pydocs::DOC_ImageStack_clear_from_gpu);
}

#endif /* Py_PYTHON_H */

} /* namespace search */
