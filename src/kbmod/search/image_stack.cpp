#include "image_stack.h"


namespace search {
ImageStack::ImageStack(const std::vector<std::string>& filenames, const std::vector<PSF>& psfs) {
    verbose = true;
    images = std::vector<LayeredImage>();
    load_images(filenames, psfs);

    global_mask = RawImage(get_width(), get_height());
    global_mask.set_all_pix(0.0);
}

ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) {
    verbose = true;
    images = imgs;

    global_mask = RawImage(get_width(), get_height());
    global_mask.set_all_pix(0.0);
}

void ImageStack::load_images(const std::vector<std::string>& filenames, const std::vector<PSF>& psfs) {
    const int num_files = filenames.size();
    if (num_files == 0) {
        std::cout << "No files provided"
                  << "\n";
    }

    if (psfs.size() != num_files) throw std::runtime_error("Mismatched PSF array in ImageStack creation.");

    // Load images from file
    for (int i = 0; i < num_files; ++i) {
        images.push_back(LayeredImage(filenames[i], psfs[i]));
        if (verbose) std::cout << "." << std::flush;
    }
    if (verbose) std::cout << "\n";
}

LayeredImage& ImageStack::get_single_image(int index) {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index];
}

float ImageStack::get_obstime(int index) const {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index].get_obstime();
}

float ImageStack::get_zeroed_time(int index) const {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index].get_obstime() - images[0].get_obstime();
}

std::vector<float> ImageStack::build_zeroed_times() const {
    std::vector<float> zeroed_times = std::vector<float>();
    if (images.size() > 0) {
        float t0 = images[0].get_obstime();
        for (auto& i : images) {
            zeroed_times.push_back(i.get_obstime() - t0);
        }
    }
    return zeroed_times;
}

void ImageStack::convolve_psf() {
    for (auto& i : images) i.convolve_psf();
}

void ImageStack::save_global_mask(const std::string& path) { global_mask.save_to_file(path); }

void ImageStack::save_images(const std::string& path) {
    for (auto& i : images) i.save_layers(path);
}

const RawImage& ImageStack::get_global_mask() const { return global_mask; }

void ImageStack::apply_mask_flags(int flags, const std::vector<int>& exceptions) {
    for (auto& i : images) {
        i.apply_mask_flags(flags, exceptions);
    }
}

void ImageStack::apply_global_mask(int flags, int threshold) {
    create_global_mask(flags, threshold);
    for (auto& i : images) {
        i.apply_global_mask(global_mask);
    }
}

void ImageStack::apply_mask_threshold(float thresh) {
    for (auto& i : images) i.apply_mask_threshold(thresh);
}

void ImageStack::grow_mask(int steps) {
    for (auto& i : images) i.grow_mask(steps);
}

void ImageStack::create_global_mask(int flags, int threshold) {
    int npixels = get_npixels();

    // For each pixel count the number of images where it is masked.
    std::vector<int> counts(npixels, 0);
    for (unsigned int img = 0; img < images.size(); ++img) {
        float* imgMask = images[img].get_mask().data();
        // Count the number of times a pixel has any of the flags
        for (unsigned int pixel = 0; pixel < npixels; ++pixel) {
            if ((flags & static_cast<int>(imgMask[pixel])) != 0) counts[pixel]++;
        }
    }

    // Set all pixels below threshold to 0 and all above to 1
    float* global_m = global_mask.data();
    for (unsigned int p = 0; p < npixels; ++p) {
        global_m[p] = counts[p] < threshold ? 0.0 : 1.0;
    }
}

#ifdef Py_PYTHON_H
static void image_stack_bindings(py::module& m) {
    using is = search::ImageStack;
    using li = search::LayeredImage;
    using pf = search::PSF;

    py::class_<is>(m, "ImageStack", pydocs::DOC_ImageStack)
            .def(py::init<std::vector<std::string>, std::vector<pf>>())
            .def(py::init<std::vector<li>>())
            .def("get_images", &is::get_images, pydocs::DOC_ImageStack_get_images)
            .def("get_single_image", &is::get_single_image, py::return_value_policy::reference_internal,
                 pydocs::DOC_ImageStack_get_single_image)
            .def("get_obstime", &is::get_obstime, pydocs::DOC_ImageStack_get_obstime)
            .def("get_zeroed_time", &is::get_zeroed_time, pydocs::DOC_ImageStack_get_zeroed_time)
            .def("build_zeroed_times", &is::build_zeroed_times, pydocs::DOC_ImageStack_build_zeroed_times)
            .def("img_count", &is::img_count, pydocs::DOC_ImageStack_img_count)
            .def("apply_mask_flags", &is::apply_mask_flags, pydocs::DOC_ImageStack_apply_mask_flags)
            .def("apply_mask_threshold", &is::apply_mask_threshold,
                 pydocs::DOC_ImageStack_apply_mask_threshold)
            .def("apply_global_mask", &is::apply_global_mask, pydocs::DOC_ImageStack_apply_global_mask)
            .def("grow_mask", &is::grow_mask, pydocs::DOC_ImageStack_grow_mask)
            .def("save_global_mask", &is::save_global_mask, pydocs::DOC_ImageStack_save_global_mask)
            .def("save_images", &is::save_images, pydocs::DOC_ImageStack_save_images)
            .def("get_global_mask", &is::get_global_mask, pydocs::DOC_ImageStack_get_global_mask)
            .def("convolve_psf", &is::convolve_psf, pydocs::DOC_ImageStack_convolve_psf)
            .def("get_width", &is::get_width, pydocs::DOC_ImageStack_get_width)
            .def("get_height", &is::get_height, pydocs::DOC_ImageStack_get_height)
            .def("get_npixels", &is::get_npixels, pydocs::DOC_ImageStack_get_npixels);
}

#endif /* Py_PYTHON_H */

} /* namespace search */
