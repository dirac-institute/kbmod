#include "image_stack.h"

#include <algorithm>
#include "logging.h"

namespace search {

ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) {
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Constructing ImageStack with " + std::to_string(imgs.size()) + " images.");
    images = imgs;

    // No data on GPU unless specifically transferred.
    data_on_gpu = false;
}

virtual ImageStack::~ImageStack() { clear_from_gpu(); }

LayeredImage& ImageStack::get_single_image(int index) {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index];
}

double ImageStack::get_obstime(int index) const {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index].get_obstime();
}

double ImageStack::get_zeroed_time(int index) const {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
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
              [](LayeredImage a, LayeredImage b) { return a.get_obstime() < b.get_obstime(); });
}

void ImageStack::convolve_psf() {
    if (data_on_gpu) throw std::runtime_error("Cannot modify images while on GPU");
    for (auto& i : images) i.convolve_psf();
}

void ImageStack::copy_to_gpu() {
    if (data_on_gpu) return;  // Nothing to do

    // Move the time data to the GPU.
    unsigned num_times = img_count();
    gpu_time_array.resize(num_times);
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Copying times to GPU: " + std::to_string(gpu_time_array.get_size()) + " items, " +
                    std::to_string(gpu_time_array.get_memory_size()) + " bytes");

    std::vector<double> image_times = stack.build_zeroed_times();
    gpu_time_array.copy_vector_to_gpu(image_times);
    if (!gpu_time_array.on_gpu()) throw std::runtime_error("Failed to copy times to GPU.");

    // Move the image data to the GPU.
    unsigned height = get_height();
    unsigned width = get_width();
    unsigned num_pixels = height * width * num_times;
    gpu_image_array.resize(num_pixels);
    logging::getLogger("kbmod.search.image_stack")
            ->debug("Copying images to GPU: " + std::to_string(gpu_image_array.get_size()) + " items, " +
                    std::to_string(gpu_image_array.get_memory_size()) + " bytes");

    std::vector<float> image_data(num_pixels);
    unsigned index = 0;
    for (unsigned t = 0; t < num_images; ++t) {
        const Image& current_img = get_single_image(t).get_science().get_image();
        for (unsigned i = 0; i < num_images; ++i) {
            for (unsigned j = 0; j < num_images; ++j) {
                image_data[index] = current_img(i, j);
                ++index;
            }
        }
    }
    gpu_image_array.copy_vector_to_gpu(image_data);
    if (!gpu_image_array.on_gpu()) throw std::runtime_error("Failed to copy images to GPU.");

    // Mark the data as copied.
    data_on_gpu = true;
}

void ImageStack::clear_from_gpu() {
    if (!data_on_gpu) return;  // Nothing to do

    logging::getLogger("kbmod.search.image_stack")
            ->debug("Freeing images on GPU: " + std::to_string(gpu_image_array.get_size()) + " items, " +
                    std::to_string(gpu_image_array.get_memory_size()) + " bytes");
    gpu_image_array.free_gpu_memory();

    logging::getLogger("kbmod.search.image_stack")
            ->debug("Freeing times on GPU: " + std::to_string(gpu_time_array.get_size()) + " items, " +
                    std::to_string(gpu_time_array.get_memory_size()) + " bytes");
    gpu_time_array.free_gpu_memory();
}

RawImage ImageStack::make_global_mask(int flags, int threshold) {
    int npixels = get_npixels();

    // Start with an empty global mask.
    RawImage global_mask = RawImage(get_width(), get_height());
    global_mask.set_all(0.0);

    // For each pixel count the number of images where it is masked.
    std::vector<int> counts(npixels, 0);
    for (unsigned int img = 0; img < images.size(); ++img) {
        auto imgMask = images[img].get_mask().get_image().reshaped();

        // Count the number of times a pixel has any of the given flags
        for (unsigned int pixel = 0; pixel < npixels; ++pixel) {
            if ((flags & static_cast<int>(imgMask[pixel])) != 0) counts[pixel]++;
        }
    }

    // Set all pixels below threshold to 0 and all above to 1
    auto global_m = global_mask.get_image().reshaped();
    for (unsigned int p = 0; p < npixels; ++p) {
        global_m[p] = counts[p] < threshold ? 0.0 : 1.0;
    }

    return global_mask;
}

#ifdef Py_PYTHON_H
static void image_stack_bindings(py::module& m) {
    using is = search::ImageStack;
    using li = search::LayeredImage;
    using pf = search::PSF;

    py::class_<is>(m, "ImageStack", pydocs::DOC_ImageStack)
            .def(py::init<std::vector<li>>())
            .def_property_readonly("on_gpu", &is::on_gpu, pydocs::DOC_ImageStack_on_gpu)
            .def("get_images", &is::get_images, pydocs::DOC_ImageStack_get_images)
            .def("get_single_image", &is::get_single_image, py::return_value_policy::reference_internal,
                 pydocs::DOC_ImageStack_get_single_image)
            .def("get_obstime", &is::get_obstime, pydocs::DOC_ImageStack_get_obstime)
            .def("get_zeroed_time", &is::get_zeroed_time, pydocs::DOC_ImageStack_get_zeroed_time)
            .def("build_zeroed_times", &is::build_zeroed_times, pydocs::DOC_ImageStack_build_zeroed_times)
            .def("sort_by_time", &is::sort_by_time, pydocs::DOC_ImageStack_sort_by_time)
            .def("img_count", &is::img_count, pydocs::DOC_ImageStack_img_count)
            .def("make_global_mask", &is::make_global_mask, pydocs::DOC_ImageStack_make_global_mask)
            .def("convolve_psf", &is::convolve_psf, pydocs::DOC_ImageStack_convolve_psf)
            .def("get_width", &is::get_width, pydocs::DOC_ImageStack_get_width)
            .def("get_height", &is::get_height, pydocs::DOC_ImageStack_get_height)
            .def("get_npixels", &is::get_npixels, pydocs::DOC_ImageStack_get_npixels)
            .def("copy_to_gpu", &is::copy_to_gpu, pydocs::DOC_ImageStack_copy_to_gpu)
            .def("clear_from_gpu", &is::clear_from_gpu, pydocs::DOC_ImageStack_clear_from_gpu);
}

#endif /* Py_PYTHON_H */

} /* namespace search */
