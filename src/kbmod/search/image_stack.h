#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <vector>
#include <dirent.h>
#include <string>
#include <list>
#include <iostream>
#include <stdexcept>

#include "gpu_array.h"
#include "kernel_testing_helpers.h"
#include "layered_image.h"
#include "pydocs/image_stack_docs.h"


namespace search {
class ImageStack {
public:
    ImageStack();
    ImageStack(const std::vector<LayeredImage>& imgs);

    // Disallow copying and assignment to avoid accidental huge memory costs
    // or invalid GPU memory pointers.
    ImageStack(ImageStack&) = delete;
    ImageStack(const ImageStack&) = delete;
    ImageStack& operator=(ImageStack&) = delete;
    ImageStack& operator=(const ImageStack&) = delete;

    // Simple getters.
    unsigned img_count() const { return images.size(); }
    unsigned get_width() const { return width; }
    unsigned get_height() const { return height; }
    uint64_t get_npixels() const { return static_cast<uint64_t>(width) * static_cast<uint64_t>(height); }
    uint64_t get_total_pixels() const { return get_npixels() * images.size(); }
    std::vector<LayeredImage>& get_images() { return images; }
    LayeredImage& get_single_image(int index);

    // Functions for setting or appending a single LayeredImage. If force_move is true,
    // then the code uses move semantics and destroys the input object.
    void set_single_image(int index, LayeredImage& img, bool force_move = false);
    void append_image(LayeredImage& img, bool force_move = false);

    // Functions for getting or using times.
    double get_obstime(int index) const;
    double get_zeroed_time(int index) const;
    std::vector<double> build_zeroed_times() const;  // Linear cost.
    void sort_by_time();

    virtual ~ImageStack();

    // Functions to handle transfering data to/from GPU.
    inline bool on_gpu() const { return data_on_gpu; }
    void copy_to_gpu();
    void clear_from_gpu();

    // Array access functions. For use when passing to the GPU only.
    GPUArray<float>& get_gpu_image_array() { return gpu_image_array; }
    GPUArray<double>& get_gpu_time_array() { return gpu_time_array; }

private:
    unsigned int width;
    unsigned int height;
    std::vector<LayeredImage> images;

    // Data pointers on the GPU.
    bool data_on_gpu;
    GPUArray<float> gpu_image_array;
    GPUArray<double> gpu_time_array;
};

} /* namespace search */

#endif /* IMAGESTACK_H_ */
