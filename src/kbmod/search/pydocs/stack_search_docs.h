#ifndef STACKSEARCH_DOCS
#define STACKSEARCH_DOCS

namespace pydocs {
static const auto DOC_StackSearch = R"doc(
  The data and configuration needed for KBMOD's core search. It is created using either
  a *reference* to the ``ImageStack`` or lists of science, variance, and PSF information.

  Attributes
  ----------
  num_images : `int`
      The number of images (or time steps).
  height : `int`
      The height of each image in pixels.
  width : `int`
      The width of each image in pixels.
  zeroed_times : `list`
      The times shift so the first time is at 0.0.

  Parameters
  ----------
  sci_imgs : `list`
      A list of science images as numpy arrays.
  var_imgs : `list`
      A list of variance images as numpy arrays.
  psf_kernels : `list`
      A list of PSF kernels as numpy arrays.
  zeroed_times : `list`
      A list of floating point times starting at zero.
  num_bytes : `int`
      The number of bytes to use for encoding the data. This is used
      to set the encoding level for the data copied to the GPU. The
      default value is -1, which means no encoding is done.
      The other options are 1 (uint8), 2 (uint16), and 4 (float).
  )doc";

static const auto DOC_StackSearch_search = R"doc(
  Perform the KBMOD search by evaluating a list of candidate trajectories at each 
  starting pixel in the image.  The results are stored in the ``StackSearch`` object
  and can be accessed with get_results().

  Parameters
  ----------
  search_list : `list`
      A list of Trajectory objects where each trajectory is evaluated at each starting pixel.
  on_gpu : `bool`
      Run the search on the GPU.
  )doc";

static const auto DOC_StackSearch_set_min_obs = R"doc(
  Sets the minimum number of observations for valid result.

  Parameters
  ----------
  new_value : `int`
      The minimum number of valid observations for a trajectory to be returned.
  )doc";

static const auto DOC_StackSearch_set_min_lh = R"doc(
  Sets the minimum likelihood for valid result.

  Parameters
  ----------
  new_value : `float`
      The minimum likelihood value for a trajectory to be returned.
  )doc";

static const auto DOC_StackSearch_disable_gpu_sigmag_filter = R"doc(
  Turns off the on-GPU sigma-G filtering.
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

static const auto DOC_StackSearch_set_start_bounds_x = R"doc(
  Set the starting and ending bounds in the x direction for a grid search.
  The grid search will test all pixels [x_min, x_max).

  Parameters
  ----------
  x_min : `int`
      The inclusive lower bound of the search (in pixels).
  x_max : `int`
      The exclusive upper bound of the search (in pixels).

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
      The inclusive lower bound of the search (in pixels).
  y_max : `int`
      The exclusive upper bound of the search (in pixels).

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

static const auto DOC_StackSearch_get_all_psi_phi_curves = R"doc(
  Return a single matrix with both the psi and phi curves. Each
  row corresponds to a single trajectory and the columns hold
  the psi values then the phi values (in order of time).

  Parameters
  ----------
  trj : `list` of `kb.Trajectory`
      The input trajectories.

  Returns
  -------
  result : `np.ndarray`
     A shape (R, 2T) matrix where R is the number of trajectories and
     T is the number of time steps. The first T columns contain the psi
     values and the second T columns contain the phi columns.
  )doc";

static const auto DOC_StackSearch_get_number_total_results = R"doc(
  Get the total number of saved results.

  Returns
  -------
  result : `int`
      The number of saved results.
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

static const auto DOC_StackSearch_get_all_results = R"doc(
  Get a reference to the full list of results.

  Returns
  -------
  results : `List`
      A list of ``Trajectory`` objects for the cached results.
  )doc";

static const auto DOC_StackSearch_compute_max_results = R"doc(
  Compute the maximum number of results according to the x, y bounds and the results per pixel.

  Returns
  -------
  max_results : `int`
      The maximum number of results that a search will return.
  )doc";

static const auto DOC_StackSearch_set_results = R"doc(
  Set the cached results. Used for testing.

  Parameters
  ----------
  new_results : `List`
      The list of results (``Trajectory`` objects) to store.
  )doc";

static const auto DOC_StackSearch_clear_results = R"doc(
  Clear the saved results.
  )doc";

static const auto DOC_StackSearch_evaluate_single_trajectory = R"doc(
  Performs the evaluation of a single Trajectory object. Modifies the 
  trajectory object in-place to add the statistics.

  Notes
  -----
  Runs on the CPU.

  Parameters
  ----------
  trj : `kb.Trajectory`
      The trjactory to evaluate.
  use_kernel : `bool`
      Use the kernel code for evaluation. This requires the code is compiled with
      the nvidia libraries, but performs the exact same computations as on GPU.                      
   )doc";

static const auto DOC_StackSearch_search_linear_trajectory = R"doc(
  Performs the evaluation of a linear trajectory in pixel space.

  Notes
  -----
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
  use_kernel : `bool`
      Use the kernel code for evaluation. This requires the code is compiled with
      the nvidia libraries, but performs the exact same computations as on GPU.

  Returns
  -------
  result : `kb.Trajectory`
      The trajectory object with statistics set.
   )doc";

}  // namespace pydocs
#endif /* STACKSEARCH_DOCS */
