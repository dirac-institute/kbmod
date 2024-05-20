#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <vector>
#include <dirent.h>
#include <string>
#include <list>
#include <iostream>
#include <stdexcept>

#include "gpu_array.h"
#include "layered_image.h"
#include "pydocs/image_stack_docs.h"

namespace search {
class ImageStack {
public:
    ImageStack(const std::vector<LayeredImage>& imgs);

    // Disallow copying and assignment to avoid accidental huge memory costs
    // or invalid GPU memory pointers.
    ImageStack(ImageStack&) = delete;
    ImageStack(const ImageStack&) = delete;
    ImageStack& operator=(ImageStack&) = delete;
    ImageStack& operator=(const ImageStack&) = delete;

    // Simple getters.
    unsigned img_count() const { return images.size(); }
    unsigned get_width() const { return images.size() > 0 ? images[0].get_width() : 0; }
    unsigned get_height() const { return images.size() > 0 ? images[0].get_height() : 0; }
    unsigned get_npixels() const { return images.size() > 0 ? images[0].get_npixels() : 0; }
    std::vector<LayeredImage>& get_images() { return images; }
    LayeredImage& get_single_image(int index);

    // Functions for getting or using times.
    double get_obstime(int index) const;
    double get_zeroed_time(int index) const;
    std::vector<double> build_zeroed_times() const;  // Linear cost.
    void sort_by_time();

    void convolve_psf();

    // Make and return a global mask.
    RawImage make_global_mask(int flags, int threshold);

    virtual ~ImageStack();

    // Functions to handle transfering data to/from GPU.
    inline bool on_gpu() const { return data_on_gpu; }
    void copy_to_gpu();
    void clear_from_gpu();

    // Array access functions. For use when passing to the GPU only.
    GPUArray<float>& get_gpu_image_array() { return gpu_image_array; }
    GPUArray<double>& get_gpu_time_array() { return gpu_time_array; }

private:
    std::vector<LayeredImage> images;

    // Data pointers on the GPU.
    bool data_on_gpu;
    GPUArray<float> gpu_image_array;
    GPUArray<double> gpu_time_array;
};

} /* namespace search */

#endif /* IMAGESTACK_H_ */
