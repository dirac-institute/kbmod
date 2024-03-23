#ifndef TRAJECTORY_LIST_DOCS_
#define TRAJECTORY_LIST_DOCS_

namespace pydocs {

static const auto DOC_TrajectoryList = R"doc(
  A list of trajectories that can be transferred between CPU or GPU.
  )doc";

static const auto DOC_TrajectoryList_on_gpu = R"doc(
  Whether the data currently resides on the GPU (True) or CPU (False)
  )doc";

static const auto DOC_TrajectoryList_get_size = R"doc(
  Return the size of the list.
  )doc";

static const auto DOC_TrajectoryList_get_trajectory = R"doc(
  Get a reference trajectory from the list. The data must reside on the CPU.

  Parameters
  ----------
  index : `int`
      The index of the entry.

  Returns
  -------
  trj : `Trajectory`
      The corresponding Trajectory object.

  Raises
  ------
  Raises a ``RuntimeError`` if the index is invalid or the data currently resides
  on the GPU.
  )doc";

static const auto DOC_TrajectoryList_set_trajectory = R"doc(
  Set a trajectory in the list. The data must reside on the CPU.

  Parameters
  ----------
  index : `int`
      The index of the entry.
  new_value : `Trajectory`
      The corresponding Trajectory object.

  Raises
  ------
  Raises a ``RuntimeError`` if the index is invalid or the data currently resides
  on the GPU.
  )doc";

static const auto DOC_TrajectoryList_set_trajectories = R"doc(
  Set an entire list of trajectories. Resizes the array to match the given
  input. The data must reside on the CPU.

  Parameters
  ----------
  new_values : `list`
      A list of `Trajectory` objects.

  Raises
  ------
  Raises a ``RuntimeError`` if the index is invalid or the data currently resides
  on the GPU.
  )doc";
 
      
static const auto DOC_TrajectoryList_get_list = R"doc(
  Return the full list of trajectories. The data must reside on the CPU.

  Returns
  -------
  result : `list`
      The list of trajectories

  Raises
  ------
  Raises a ``RuntimeError`` if the data currently resides on the GPU.
  )doc";

static const auto DOC_TrajectoryList_get_batch = R"doc(
  Return a batch of results. The data must reside on the CPU.

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
  ``RunTimeError`` if start < 0 or count <= 0 or if the data is on GPU.
  )doc";

static const auto DOC_TrajectoryList_resize = R"doc(
  Forcibly resize the array. If the size is decreased, the extra entries
  are dropped from the back. If the size is increased, extra (blank)
  trajectories are added to the back.
      
  The data must reside on the CPU.

  Parameters
  ----------
  new_size : `int`
      The new size of the list.

  Raises
  ------
  ``RunTimeError`` if new_size < 0 or data is on GPU.
  )doc";

static const auto DOC_TrajectoryList_move_to_cpu = R"doc(
  Move the data from GPU to CPU. If the data is already on the CPU
  this is a no-op.

  Raises
  ------
  Raises a ``RuntimeError`` if invalid state encountered.
  )doc";

static const auto DOC_TrajectoryList_move_to_gpu = R"doc(
  Move the data from CPU to GPU. If the data is already on the GPU
  this is a no-op.

  Raises
  ------
  Raises a ``RuntimeError`` if invalid state encountered.
  )doc";

static const auto DOC_TrajectoryList_sort_by_likelihood = R"doc(
  Sort the data in order of decreasing likelihood. The data must reside on the CPU.

  Raises
  ------
  Raises a ``RuntimeError`` the data is on GPU.
  )doc";

static const auto DOC_TrajectoryList_sort_by_obs_count = R"doc(
  Sort the data in order of decreasing obs_count. The data must reside on the CPU.

  Raises
  ------
  Raises a ``RuntimeError`` the data is on GPU.
  )doc";

static const auto DOC_TrajectoryList_filter_by_likelihood = R"doc(
  Sort the data in order of decreasing likelihood and drop everything less than
  a given threshold. The data must reside on the CPU.

  Parameters
  ----------
  min_likelihood : `float`
      The threshold on minimum likelihood.
      
  Raises
  ------
  Raises a ``RuntimeError`` the data is on GPU.
  )doc";

static const auto DOC_TrajectoryList_filter_by_obs_count = R"doc(
  Sort the data in order of decreasing obs_count and drop everything less than
  a given threshold. The data must reside on the CPU.

  Parameters
  ----------
  min_obs_count : `int`
      The threshold on minimum number of observations.
      
  Raises
  ------
  Raises a ``RuntimeError`` the data is on GPU.
  )doc";

static const auto DOC_TrajectoryList_filter_by_valid = R"doc(
  Filter out all trajectories with the ``valid`` attribute set to ``False``.
  Ordering is not preserved. The data must reside on the CPU.
      
  Raises
  ------
  Raises a ``RuntimeError`` the data is on GPU.
  )doc";

}  // namespace pydocs

#endif /* TRAJECTORY_LIST_DOCS_ */
