/*
 * ImageStack.cpp
 *
 *  Created on: Jun 22, 2017
 *      Author: kbmod-usr
 */

#include "ImageStack.h"

namespace search {

  ImageStack::ImageStack(const std::vector<std::string>& filenames, const std::vector<PSF>& psfs) {
    verbose = true;
    resetImages();
    loadImages(filenames, psfs);
    extractImageTimes();
    setTimeOrigin();
    global_mask = RawImage(getWidth(), getHeight());
    global_mask.set_all_pix(0.0);
  }

  ImageStack::ImageStack(const std::vector<LayeredImage>& imgs) {
    verbose = true;
    images = imgs;
    extractImageTimes();
    setTimeOrigin();
    global_mask = RawImage(getWidth(), getHeight());
    global_mask.set_all_pix(0.0);
  }

  void ImageStack::loadImages(const std::vector<std::string>& filenames,
                              const std::vector<PSF>& psfs) {
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

  void ImageStack::extractImageTimes() {
    // Load image times
    image_times = std::vector<float>();
    for (auto& i : images) {
      image_times.push_back(float(i.get_obstime()));
    }
  }

  void ImageStack::setTimeOrigin() {
    // Set beginning time to 0.0
    double initial_time = image_times[0];
    for (auto& t : image_times) t = t - initial_time;
  }

  LayeredImage& ImageStack::getSingleImage(int index) {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    return images[index];
  }

  void ImageStack::setSingleImage(int index, LayeredImage& img) {
    if (index < 0 || index > images.size()) throw std::out_of_range("ImageStack index out of bounds.");
    images[index] = img;
  }

  void ImageStack::setTimes(const std::vector<float>& times) {
    if (times.size() != imgCount())
      throw std::runtime_error(
                               "List of times provided"
                               " does not match the number of images!");
    image_times = times;
    setTimeOrigin();
  }

  void ImageStack::resetImages() { images = std::vector<LayeredImage>(); }

  void ImageStack::convolvePSF() {
    for (auto& i : images) i.convolve_psf();
  }

  void ImageStack::saveGlobalMask(const std::string& path) { global_mask.save_to_file(path); }

  void ImageStack::saveImages(const std::string& path) {
    for (auto& i : images) i.save_layers(path);
  }

  const RawImage& ImageStack::getGlobalMask() const { return global_mask; }

  void ImageStack::apply_maskFlags(int flags, const std::vector<int>& exceptions) {
    for (auto& i : images) {
      i.apply_mask_flags(flags, exceptions);
    }
  }

  void ImageStack::applyGlobalMask(int flags, int threshold) {
    createGlobalMask(flags, threshold);
    for (auto& i : images) {
      i.applyGlobalMask(global_mask);
    }
  }

  void ImageStack::apply_maskThreshold(float thresh) {
    for (auto& i : images) i.apply_mask_threshold(thresh);
  }

  void ImageStack::growMask(int steps) {
    for (auto& i : images) i.grow_mask(steps);
  }

  void ImageStack::createGlobalMask(int flags, int threshold) {
    int npixels = getNPixels();

    // For each pixel count the number of images where it is masked.
    std::vector<int> counts(npixels, 0);
    for (unsigned int img = 0; img < images.size(); ++img) {
      float* imgMask = images[img].getMDataRef();
      // Count the number of times a pixel has any of the flags
      for (unsigned int pixel = 0; pixel < npixels; ++pixel) {
        if ((flags & static_cast<int>(imgMask[pixel])) != 0) counts[pixel]++;
      }
    }

    // Set all pixels below threshold to 0 and all above to 1
    float* global_m = global_mask.getDataRef();
    for (unsigned int p = 0; p < npixels; ++p) {
      global_m[p] = counts[p] < threshold ? 0.0 : 1.0;
    }
  }

} /* namespace search */
