#ifndef IMAGESTACK_H_
#define IMAGESTACK_H_

#include <vector>
#include <dirent.h>
#include <string>
#include <list>
#include <iostream>
#include <stdexcept>

#include "layered_image.h"
#include "pydocs/image_stack_docs.h"

namespace search {
class ImageStack {
public:
    ImageStack(const std::vector<LayeredImage>& imgs);

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

    virtual ~ImageStack(){};

private:
    std::vector<LayeredImage> images;
};

} /* namespace search */

#endif /* IMAGESTACK_H_ */
