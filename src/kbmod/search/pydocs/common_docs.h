#ifndef COMMON_DOCS
#define COMMON_DOCS

namespace pydocs {
  static const auto DOC_Trajectory = R"doc(
  A trajectory structure holding basic information about potential results.

  Attributes
  ----------
    x : `float`
        x coordinate of the origin (what?)
    y : `float`
        y coordinate of the origin (what?)
    vx : `float`
        x component of the velocity, as projected on the image
    vy : `float`
        y component of the velocity, as projected on the image
    lh : `float`
        Likelihood (accumulated?)
    flux : `float`
        Flux (accumulated?)
    obs_count : `int`
        Number of observations trajectory was seen in.
  )doc";

  static const auto DOC_PixelPos = R"doc(
  todo
  )doc";

  static const auto DOC_ImageMoments = R"doc(
  todo
  )doc";

  static const auto DOC_StampParameters = R"doc(
  todo
  ")doc";

  static const auto DOC_BaryCorrection =  R"doc(
  todo
  ")doc";

} // namespace pydocs

#endif /* COMMON_DOCS */
