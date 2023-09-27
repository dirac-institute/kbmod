#ifndef IMAGESTACK_DOCS
#define IMAGESTACK_DOCS

namespace pydocs {
  static const auto DOC_ImageStack = R"doc(
  todo
  )doc";
      
  static const auto DOC_ImageStack_get_images = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_img_count = R"doc(
  Returns the number of images in the stack.
  )doc";
      
  static const auto DOC_ImageStack_get_single_image = R"doc(
  Returns a single LayeredImage for a given index.
  )doc";
      
  static const auto DOC_ImageStack_get_obstime = R"doc(
  Returns a single image's observation time in MJD.
  )doc";
    
  static const auto DOC_ImageStack_get_zeroed_time = R"doc(
  Returns a single image's observation time relative to that
  of the first image.
  )doc";

  static const auto DOC_ImageStack_build_zeroed_times = R"doc(
  Construct an array of time differentials between each image
  in the stack and the first image.
  ")doc";
    
  static const auto DOC_ImageStack_apply_mask_flags = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_apply_mask_threshold = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_apply_global_mask = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_grow_mask = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_save_global_mask = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_save_images = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_get_global_mask = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_convolve_psf = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_get_width = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_get_height = R"doc(
  todo
  )doc";

  static const auto DOC_ImageStack_get_npixels = R"doc(
  todo
  )doc";

} /* pydocs */

#endif /* IMAGESTACK_DOCS */
