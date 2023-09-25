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
    ImageStack(const std::vector<std::string>& filenames, const std::vector<PSF>& psfs);
    ImageStack(const std::vector<LayeredImage>& imgs);

    // Simple getters.
    unsigned img_count() const { return images.size(); }
    unsigned get_width() const { return images.size() > 0 ? images[0].get_width() : 0; }
    unsigned get_height() const { return images.size() > 0 ? images[0].get_height() : 0; }
    unsigned get_npixels() const { return images.size() > 0 ? images[0].get_npixels() : 0; }
    std::vector<LayeredImage>& get_images() { return images; }
    const std::vector<float>& get_times() const { return image_times; }
    float* get_timesDataRef() { return image_times.data(); }
    LayeredImage& get_single_image(int index);

    // Simple setters.
    void set_times(const std::vector<float>& times);
    void reset_images();
    void set_single_image(int index, LayeredImage& img);

    // Apply makes to all the images.
    void apply_global_mask(int flags, int threshold);
    void apply_mask_flags(int flags, const std::vector<int>& exceptions);
    void apply_mask_threshold(float thresh);
    void grow_mask(int steps);
    const RawImage& get_global_mask() const;

    void convolve_psf();

    // Save data to files.
    void save_global_mask(const std::string& path);
    void save_images(const std::string& path);

    virtual ~ImageStack(){};

  private:
    void load_images(const std::vector<std::string>& filenames, const std::vector<PSF>& psfs);
    void extract_image_times();
    void set_time_origin();
    void create_global_mask(int flags, int threshold);
    std::vector<LayeredImage> images;
    RawImage global_mask;
    std::vector<float> image_times;
    bool verbose;
  };

} /* namespace search */

#endif /* IMAGESTACK_H_ */
