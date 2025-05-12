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
      A zero shifted time in days.
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
      A zero shifted time in days.
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
      A zero shifted time in days.
      
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
      A zero shifted time in days.

  Returns
  -------
  `int`
     The predicted row index.
  )doc";

}  // namespace pydocs

#endif /* COMMON_DOCS */
