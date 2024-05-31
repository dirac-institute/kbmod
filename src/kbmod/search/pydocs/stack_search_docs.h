#ifndef STACKSEARCH_DOCS
#define STACHSEARCH_DOCS

namespace pydocs {
static const auto DOC_StackSearch = R"doc(
  The data and configuration needed for KBMOD's core search. It is created
  using a *reference* to the ``ImageStack``. The underlying ``ImageStack``
  must exist for the life of the ``StackSearch`` object's life.
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
  Enable on-GPU sigma-G filtering.

  Parameters
  ----------
  percentiles : `list`
      A length 2 list of percentiles (between 0.0 and 1.0). Example [0.25, 0.75].
  sigmag_coeff : `float`
      The sigma-G coefficient corresponding to the percentiles. This can
      be computed via SigmaGClipping.find_sigma_g_coeff().
  min_lh : `float`
      The minimum likelihood for a result to be accepted.

  Raises
  ------
  Raises a ``RunTimeError`` if invalid values are provided.
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

  Raises
  ------
  Raises a ``RunTimeError`` if invalid bounds are provided (x_max > x_min).
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

  Raises
  ------
  Raises a ``RunTimeError`` if invalid bounds are provided (x_max > x_min).
  )doc";

static const auto DOC_StackSearch_set_results_per_pixel = R"doc(
  Set the maximum number of results per pixel returns by a search.

  Parameters
  ----------
  new_value : `int`
      The new number of results per pixel.

  Raises
  ------
  Raises a ``RunTimeError`` if an invalid value is provided (new_value <= 0).
  )doc";

static const auto DOC_StackSearch_get_num_images = R"doc(
  Returns the number of images to process.
  )doc";

static const auto DOC_StackSearch_get_image_width = R"doc(
  Returns the width of the images in pixels.
  )doc";

static const auto DOC_StackSearch_get_image_height = R"doc(
  Returns the height of the images in pixels.
  )doc";

static const auto DOC_StackSearch_get_image_npixels = R"doc(
  Returns the number of pixels for each image.
  )doc";

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
  Get a batch of cached results.

  Parameters
  ----------
  start : `int`
      The starting index of the results to retrieve. Returns
      an empty list if start is past the end of the cache.
  count : `int`
      The maximum number of results to retrieve. Returns fewer
      results if there are not enough in the cache.

  Returns
  -------
  results : `List`
      A list of ``Trajectory`` objects for the cached results.

  Raises
  ------
  ``RunTimeError`` if start < 0 or count <= 0.
  )doc";

static const auto DOC_StackSearch_prepare_batch_search = R"doc(
  Prepare the search for a batch of trajectories.

  Parameters
  ----------
  search_list : `List`
      A list of ``Trajectory`` objects to search.
  min_observations : `int`
      The minimum number of observations for a trajectory to be considered.
  )doc";

static const auto DOC_StackSearch_compute_max_results = R"doc(
  Compute the maximum number of results according to the x, y bounds and the results per pixel.

  Returns
  -------
  max_results : `int`
      The maximum number of results that a search will return.
  )doc";

static const auto DOC_StackSearch_search_single_batch = R"doc(
  Perform a search on the given trajectories for the current batch.
  Batch is defined by the parameters set `set_start_bounds_x` & `set_start_bounds_y`.

  Returns
  -------
  results : `List`
      A list of ``Trajectory`` search results
  )doc";

static const auto DOC_StackSearch_finish_search = R"doc(
  Clears memory used for the batch search.

  This method should be called after a batch search is completed to ensure that any resources allocated during the search are properly freed.

  Returns
  -------
  None
  )doc";

static const auto DOC_StackSearch_set_results = R"doc(
  Set the cached results. Used for testing.

  Parameters
  ----------
  new_results : `List`
      The list of results to store.
  )doc";

static const auto DOC_StackSearch_clear_results = R"doc(
  Clear the cached results.
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
