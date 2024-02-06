#ifndef STACKSEARCH_DOCS
#define STACHSEARCH_DOCS

namespace pydocs {
static const auto DOC_StackSearch = R"doc(
  todo
  )doc";

static const auto DOC_StackSearch_search = R"doc(
  todo
  )doc";

static const auto DOC_StackSearch_set_min_obs = R"doc(
  Sets the minimum number of observations for valid result.

  Parameters
  ----------
  new_value : `int`
      The minimum number of observations for a trajectory to be returned.
  )doc";

static const auto DOC_StackSearch_set_min_lh = R"doc(
  Sets the minimum likelihood for valid result.

  Parameters
  ----------
  new_value : `float`
      The minimum likelihood value for a trajectory to be returned.
  )doc";


static const auto DOC_StackSearch_enable_gpu_sigmag_filter = R"doc(
  todo
  )doc";

static const auto DOC_StackSearch_enable_gpu_encoding = R"doc(
  Set the encoding level for the data copied to the GPU.
      1 = uint8
      2 = uint16
      4 or -1 = float

  Parameters
  ----------
  encode_num_bytes : `int`
      The number of bytes to use for encoding the data.
  )doc";

static const auto DOC_StackSearch_set_start_bounds_x = R"doc(
  Set the starting and ending bounds in the x direction for a grid search.
  The grid search will test all pixels [x_min, x_max).

  Parameters
  ----------
  x_min : `int`
      The inclusive lower bound of the search.
  x_max : `int`
      The exclusive upper bound of the search.
  )doc";

static const auto DOC_StackSearch_set_start_bounds_y = R"doc(
  Set the starting and ending bounds in the y direction for a grid search.
  The grid search will test all pixels [y_min, y_max).

  Parameters
  ----------
  y_min : `int`
      The inclusive lower bound of the search.
  y_max : `int`
      The exclusive upper bound of the search.
  )doc";

static const auto DOC_StackSearch_set_debug = R"doc(
  Set whether to dislpay debug output.

  Parameters
  ----------
  d : `bool`
      Set to ``True`` to turn on debug output and ``False`` to turn it off.
  )doc";

static const auto DOC_StackSearch_get_num_images = R"doc(
  "Returns the number of images to process.
  ")doc";

static const auto DOC_StackSearch_get_image_width = R"doc(
  "Returns the width of the images in pixels.
  ")doc";

static const auto DOC_StackSearch_get_image_height = R"doc(
  "Returns the height of the images in pixels.
  ")doc";

static const auto DOC_StackSearch_get_image_npixels = R"doc(
  "Returns the number of pixels for each image.
  ")doc";

static const auto DOC_StackSearch_get_imagestack = R"doc(
  Return the `kb.ImageStack` containing the data to search.
  )doc";

static const auto DOC_StackSearch_get_psi_curves = R"doc(
  Return the time series of psi values for a given trajectory in pixel space.

  Parameters
  ----------
  trj : `kb.Trajectory`
      The input trajectory.

  Returns
  -------
  result : `list` of `float`
     The psi values at each time step with NO_DATA replaced by 0.0.
  )doc";

static const auto DOC_StackSearch_get_phi_curves = R"doc(
  Return the time series of phi values for a given trajectory in pixel space.

  Parameters
  ----------
  trj : `kb.Trajectory`
      The input trajectory.

  Returns
  -------
  result : `list` of `float`
     The phi values at each time step with NO_DATA replaced by 0.0.
  )doc";

static const auto DOC_StackSearch_clear_psi_phi = R"doc(
  Clear the pre-computed psi and phi data.
  )doc";

static const auto DOC_StackSearch_prepare_psi_phi = R"doc(
  Compute the cached psi and phi data.
  )doc";

static const auto DOC_StackSearch_get_results = R"doc(
  todo
  )doc";

static const auto DOC_StackSearch_set_results = R"doc(
  todo
  )doc";

static const auto DOC_StackSearch_evaluate_single_trajectory = R"doc(
  Performs the evaluation of a single Trajectory object. Modifies the object
  in-place.

  Note
  ----
  Runs on the CPU, but requires CUDA compiler.

  Parameters
  ----------
  trj : `kb.Trajectory`
      The trjactory to evaluate.
   )doc";

static const auto DOC_StackSearch_search_linear_trajectory = R"doc(
  Performs the evaluation of a linear trajectory in pixel space.

  Note
  ----
  Runs on the CPU, but requires CUDA compiler.

  Parameters
  ----------
  x : `short`
      The starting x pixel of the trajectory.
  y : `short`
      The starting y pixel of the trajectory.
  vx : `float`
      The x velocity of the trajectory in pixels per day.
  vy : `float`
      The y velocity of the trajectory in pixels per day.

  Returns
  -------
  result : `kb.Trajectory`
      The trajectory object with statistics set.
   )doc";

}  // namespace pydocs
#endif /* STACKSEARCH_DOCS */
