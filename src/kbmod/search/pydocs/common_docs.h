#ifndef COMMON_DOCS
#define COMMON_DOCS

namespace pydocs {
static const auto DOC_Trajectory = R"doc(
  A structure for holding basic information about potential results
  in the form of a linear trajectory in pixel space.

  Attributes
  ----------
    x : `float`
        x coordinate of the trajectory at first time step (in pixels)
    y : `float`
        y coordinate of the trajectory at first time step (in pixels)
    vx : `float`
        x component of the velocity, as projected on the image
        (in pixels per day)
    vy : `float`
        y component of the velocity, as projected on the image
        (in pixels per day)
    lh : `float`
        The computed likelihood of all (valid) points along the trajectory.
    flux : `float`
        The computed likelihood of all (valid) points along the trajectory.
    obs_count : `int`
        The number of valid points along the trajectory.
  )doc";

static const auto DOC_Trajectory_get_x_pos = R"doc(
  Returns the predicted x position of the trajectory.

  Parameters
  ----------
  time : `float`
      A zero shifted time.
  centered : `bool`
      Shift the prediction to be at the center of the pixel
      (e.g. xp = x + vx * time + 0.5f). Default = True.

  Returns
  -------
  `float`
     The predicted x position (in pixels).
  )doc";

static const auto DOC_Trajectory_get_y_pos = R"doc(
  Returns the predicted y position of the trajectory.

  Parameters
  ----------
  time : `float`
      A zero shifted time.
  centered : `bool`
      Shift the prediction to be at the center of the pixel
      (e.g. xp = x + vx * time + 0.5f). Default = True.

  Returns
  -------
  `float`
     The predicted y position (in pixels).
  )doc";

static const auto DOC_Trajectory_get_x_index = R"doc(
  Returns the predicted x position of the trajectory as an integer
  (column) index.

  Parameters
  ----------
  time : `float`
      A zero shifted time.
      
  Returns
  -------
  `int`
     The predicted column index.
  )doc";

static const auto DOC_Trajectory_get_y_index = R"doc(
  Returns the predicted x position of the trajectory as an integer
  (row) index.

  Parameters
  ----------
  time : `float`
      A zero shifted time.

  Returns
  -------
  `int`
     The predicted row index.
  )doc";

static const auto DOC_Trajectory_is_close = R"doc(
  Checks whether a second Trajectory falls within given thresholds
  of closeness for pixel difference and velocity difference.

  Parameters
  ----------
  trj_b : `Trajectory`
      The Trajectory to test.
  pos_thresh : `float`
      The maximum separation in each of the x and y dimension (in pixels).
  vel_thresh : `float`
      The maximum separation in each of the x and y velocities (in pixels/day).

  Returns
  -------
  `bool`
      Whether the two trajectories are close.
  )doc";

static const auto DOC_ImageMoments = R"doc(
  The central moments of an image (capture how Gaussian-like an image is)

  Attributes
  ----------
  m00 : `float`
      The m00 central moment.
  m01 : `float`
      The m01 central moment.
  m10 : `float`
      The m10 central moment.
  m11 : `float`
      The m11 central moment.
  m02 : `float`
      The m02 central moment.
  m20 : `float`
      The m20 central moment.
)doc";

static const auto DOC_StampParameters = R"doc(
  Parameters for stamp generation and filtering.

  Attributes
  ----------
  radius : `int`
     The stamp radius (in pixels)
  stamp_type : `StampType`
     The co-add method to use for co-added stamps. Must be one of
     STAMP_SUM, STAMP_MEAN, or STAMP_MEDIAN.
  do_filtering : `bool`
     Indicates whether to do stamp-based filtering.
  center_thresh : `float`
     The minimum percentage of total flux at the central pixels
     for a valid coadded stamp.
  peak_offset_x : `float`
     The minimum x offset (in pixels) of the brightest location in the
     coadded stamp to filter.
  peak_offset_y : `float`
     The minimum y offset (in pixels) of the brightest location in the
     coadded stamp to filter.
  m01_limit : `float`
      The minimum m01 central moment to filter a coadded stamp.
  m10_limit : `float`
      The minimum m10 central moment to filter a coadded stamp.
  m11_limit : `float`
      The minimum m11 central moment to filter a coadded stamp.
  m02_limit : `float`
      The minimum m02 central moment to filter a coadded stamp.
  m20_limit : `float`
      The minimum m20 central moment to filter a coadded stamp.
  )doc";

}  // namespace pydocs

#endif /* COMMON_DOCS */
