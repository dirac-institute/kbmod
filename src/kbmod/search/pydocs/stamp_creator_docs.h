#ifndef STAMP_CREATOR_DOCS
#define STAMP_CREATOR_DOCS

namespace pydocs {
static const auto DOC_StampCreator = R"doc(
  A class for creating a set of stamps or a co-added stamp
  from an ImageStack and Trajectory.
  )doc";

static const auto DOC_StampCreator_create_stamps = R"doc(
  Create a vector of stamps centered on the predicted position
  of an Trajectory at different times.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  keep_no_data : `bool`
      A Boolean indicating whether to preserve NO_DATA tags or to
      replace them with 0.0.
  use_index : `list` of `bool`
      A list (vector) of Booleans indicating whether or not to use each time step.
      An empty (size=0) vector will use all time steps.
  
  Returns
  -------
  `list` of `RawImage`
      The stamps.
  )doc";

static const auto DOC_StampCreator_get_stamps = R"doc(
  Create a vector of stamps centered on the predicted position
  of an Trajectory at different times. Replaces NO_DATA with 0.0
  and returns stamps for all time steps.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  
  Returns
  -------
  `list` of `RawImage`
      The stamps.
  )doc";

static const auto DOC_StampCreator_get_median_stamp = R"doc(
  Create the median co-added stamp centered on the predicted position
  of an Trajectory at different times. Preserves NO_DATA tag.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  use_index : `list` of `bool`
      A list (vector) of Booleans indicating whether or not to use each time step.
      An empty (size=0) vector will use all time steps.

  Returns
  -------
  `RawImage`
      The co-added stamp.
  )doc";

static const auto DOC_StampCreator_get_mean_stamp = R"doc(
  Create the mean co-added stamp centered on the predicted position
  of an Trajectory at different times. Preserves NO_DATA tag.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  use_index : `list` of `bool`
      A list (vector) of Booleans indicating whether or not to use each time step.
      An empty (size=0) vector will use all time steps.

  Returns
  -------
  `RawImage`
      The co-added stamp.
  )doc";

static const auto DOC_StampCreator_get_summed_stamp = R"doc(
  Create the summed co-added stamp centered on the predicted position
  of an Trajectory at different times. Replaces NO_DATA tag with 0.0.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  use_index : `list` of `bool`
      A list (vector) of Booleans indicating whether or not to use each time step.
      An empty (size=0) vector will use all time steps.

  Returns
  -------
  `RawImage`
      The co-added stamp.
  )doc";

static const auto DOC_StampCreator_get_coadded_stamps = R"doc(
  Create a vector of co-added stamps centered on the predicted position
  of trajectories at different times.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `list` of `Trajectory`
      The list of trajectories to uses.
  use_index : `list` of `list` of `bool`
      A list of lists (vectors) of Booleans indicating whether or not to use each
      time step. use_index[i][j] indicates whether we should use timestep j of
      trajectory i. An empty (size=0) list for any trajectory will use all time
      steps for that trajectory.
  params : `StampParameters`
      The parameters for stamp generation, such as radius and co-add type.
  use_gpu : `bool`
      A Boolean indicating whether to do the co-adds on the CPU (False) or
      GPU (True).

  Returns
  -------
  `list` of `RawImage`
      The co-added stamps.

  )doc";

static const auto DOC_StampCreator_filter_stamp = R"doc(
  Filters stamps based on the given parameters.
      
  Applies the following filters: peak position, percent flux at central pixel,
  and image moments.

  Parameters
  ----------
  img : `RawImage`
      The image to test.
  params : `StampParameters`
      The parameters for stamp generation and filtering.

  Returns
  -------
  `bool`
      Whether or not to filter the stamp.
  )doc";

static const auto DOC_StampCreator_create_variance_stamps = R"doc(
  Create a vector of stamps from the variance layer centered on the
  predicted position of an Trajectory at different times.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.
  
  Returns
  -------
  `list` of `RawImage`
      The stamps.
  )doc";

static const auto DOC_StampCreator_get_variance_weighted_stamp = R"doc(
  Create a weighted-mean stamp where the weight for each pixel is 1.0 / variance.

  Parameters
  ----------
  stack : `ImageStack`
      The stack of images to use.
  trj : `Trajectory`
      The trajectory to project to each time.
  radius : `int`
      The stamp radius. Width = 2*radius+1.

  Returns
  -------
  `RawImage`
      The co-added stamp.
  )doc";

}  // namespace pydocs

#endif /* STAMP_CREATOR_DOCS */
