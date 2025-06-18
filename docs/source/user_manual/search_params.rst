Search Parameters
=================

Search parameters are set extensively via the :py:class:`~kbmod.configuration.SearchConfiguration` object. We use a custom object, instead of a standard dictionary, to both add helper functions, such as I/O and validity checking, and also to set defaults. All of the standard parameters are given default values (shown in the Table below) unless explicitly set. 


Creating a SearchConfiguration
------------------------------

There are several methods for setting these parameters. 

First, parameters can be set one-by-one from a default :py:class:`~kbmod.configuration.SearchConfiguration` object using the ``set()`` method::

    config = SearchConfiguration()
    config.set("cluster_eps", 10.0)

Second, you can provide a YAML file of the parameters using the ``config_file`` parameter::

    config = SearchConfiguration.from_file(file_path)

Third, you can pass in a dictionary mapping parameter name to parameter value::

    config = SearchConfiguration.from_dict(param_dict)

The dictionary values take precedence over all other settings, allowing you to use KBMOD as part of an internal loop over parameters.

In addition :py:class:`~kbmod.configuration.SearchConfiguration` objects are automatically saved and loaded within a :py:class:`~~kbmod.work_unit.WorkUnit`.


Configuration Parameters
------------------------

+------------------------+-----------------------------+----------------------------------------+
| **Parameter**          | **Default Value**           | **Interpretation**                     |
+------------------------+-----------------------------+----------------------------------------+
| ``clip_negative``      | False                       | An option used with sigmaG filtering,  |
|                        |                             | remove all negative values prior to    |
|                        |                             | computing the percentiles.             |
+------------------------+-----------------------------+----------------------------------------+
| ``cluster_eps``        | 20.0                        | The threshold to use for clustering    |
|                        |                             | similar results.                       |
+------------------------+-----------------------------+----------------------------------------+
| ``cluster_type``       | all                         | Types of predicted values to use when  |
|                        |                             | determining trajectories to clustered  |
|                        |                             | together, including position and       |
|                        |                             | velocities  (if do_clustering = True). |
|                        |                             | Options include: ``all``, ``position``,|
|                        |                             | ``mid_position``, and                  |
|                        |                             | ``start_end_position``                 |
+------------------------+-----------------------------+----------------------------------------+
| ``cluster_v_scale``    | 1.0                         | The weight of differences in velocity  |
|                        |                             | relative to differences in distances   |
|                        |                             | during clustering.                     |
+------------------------+-----------------------------+----------------------------------------+
| ``coadds``             | []                          | A list of additional coadds to create. |
|                        |                             | These are not used in filtering, but   |
|                        |                             | saved to columns for analysis. Can     |
|                        |                             | include: "sum", "mean", "median", and  |
|                        |                             | "weighted".                            |
|                        |                             | The filtering coadd is controlled by   |
|                        |                             | the ``stamp_type`` parameter.          |
+------------------------+-----------------------------+----------------------------------------+
| ``cpu_only``           | False                       | Perform the core search on the CPU     |
|                        |                             | (even if the GPU is available).        |
+------------------------+-----------------------------+----------------------------------------+
| ``compute_ra_dec``     | True                        | Compute and save the predicted RA and  |
|                        |                             | dec for each result at each time.      |
+------------------------+-----------------------------+----------------------------------------+
| ``debug``              | False                       | Display debugging output.              |
+------------------------+-----------------------------+----------------------------------------+
| ``do_clustering``      | True                        | Cluster the resulting trajectories to  |
|                        |                             | remove duplicates and known objects.   |
|                        |                             | See :ref:`Clustering` for more.        |
+------------------------+-----------------------------+----------------------------------------+
| ``drop_columns``       | []                          | A list of column names to skip when    |
|                        |                             | outputting results.                    |
+------------------------+-----------------------------+----------------------------------------+
| ``encode_num_bytes``   | -1                          | The number of bytes to use to encode   |
|                        |                             | ``psi`` and ``phi`` images on GPU. By  |
|                        |                             | default a ``float`` encoding is used.  |
|                        |                             | When either ``1`` or ``2``, the images |
|                        |                             | are compressed into ``unsigned int``.  |
+------------------------+-----------------------------+----------------------------------------+
| ``gpu_filter``         | False                       | Perform the filtering on the GPU. Only |
|                        |                             | ``filter_type=clipped_sigmaG``         |
|                        |                             | filtering is supported on GPU.         |
+------------------------+-----------------------------+----------------------------------------+
| ``generator_config``   | None                        | The configuration dictionary for the   |
|                        |                             | trajectory generator that will create  |
|                        |                             | the search candidates.                 |
+------------------------+-----------------------------+----------------------------------------+
| ``generate_psi_phi``   | True                        | If True, computes the psi and phi      |
|                        |                             | and saves them with the results.       |
+------------------------+-----------------------------+----------------------------------------+
| ``lh_level``           | 10.0                        | The minimum computed likelihood for an |
|                        |                             | object to be accepted.                 |
+------------------------+-----------------------------+----------------------------------------+
| ``max_results``        | 100000                      | The maximum number of results to save  |
|                        |                             | after all filtering.  The highest      |
|                        |                             | likelihood results are saved. Use -1   |
|                        |                             | to save all results.                   |
+------------------------+-----------------------------+----------------------------------------+
| ``near_dup_thresh``    | 10                          | The grid size (in pixels) for near     |
|                        |                             | duplicate pre-filtering. Use ``None``  |
|                        |                             | to skip this filtering step.           |
+------------------------+-----------------------------+----------------------------------------+
| ``nightly_coadds``     | False                       | Generate a coadd for each calendar     |
|                        |                             | date.                                  |
+------------------------+-----------------------------+----------------------------------------+
| ``num_obs``            | 10                          | The minimum number of non-masked       |
|                        |                             | observations for the object to be      |
|                        |                             | accepted. If this is greater than the  |
|                        |                             | number of the valid images or set to   |
|                        |                             | -1 then it is reduced to the number of |
|                        |                             | the valid images.                      |
+------------------------+-----------------------------+----------------------------------------+
| ``psf_val``            | 1.4                         | The value for the standard deviation of|
|                        |                             | the point spread function (PSF) in     |
|                        |                             | pixels                                 |
+------------------------+-----------------------------+----------------------------------------+
| ``result_filename``    | None                        | Full filename and path for a single    |
|                        |                             | tabular result saved as an ecsv, hdf5, |
|                        |                             | or parquet depending on the suffix.    |
+------------------------+-----------------------------+----------------------------------------+
| ``results_per_pixel``  | 8                           | The maximum number of results to       |
|                        |                             | to return for each pixel search.       |
+------------------------+-----------------------------+----------------------------------------+
| ``save_all_stamps``    | True                        | Save the individual stamps for each    |
|                        |                             | result and timestep.                   |
+------------------------+-----------------------------+----------------------------------------+
| ``separate_col_files`` | ["all_stamps"]              | A list of column names to break out    |
|                        |                             | into separate files. These files will  |
|                        |                             | be saved in the same directory as the  |
|                        |                             | main result file.                      |
+------------------------+-----------------------------+----------------------------------------+
| ``sigmaG_lims``        | [25, 75]                    | The percentiles to use in sigmaG       |
|                        |                             | filtering, if                          |
|                        |                             | ``filter_type= clipped_sigmaG``.       |
+------------------------+-----------------------------+----------------------------------------+
| ``stamp_radius``       | 10                          | Half the size of a side of a box cut   |
|                        |                             | around the predicted position when     |
|                        |                             | creating a stamp for stamp filtering   |
|                        |                             | (in pixels).                           |
+------------------------+-----------------------------+----------------------------------------+
| ``stamp_type``         | sum                         | The type of coadd to use as the main   |
|                        |                             | stamp:                                 |
|                        |                             | * ``sum`` - (default) Per pixel sum    |
|                        |                             | * ``median`` - Per pixel median        |
|                        |                             | * ``mean`` - Per pixel mean            |
|                        |                             | * ``weighted`` - Per pixel mean        |
|                        |                             | weighted by 1.0 / variance.            |
+------------------------+-----------------------------+----------------------------------------+
| ``track_filtered``     | False                       | A Boolean indicating whether to track  |
|                        |                             | the filtered trajectories. Warning     |
|                        |                             | can use a lot of memory.               |
+------------------------+-----------------------------+----------------------------------------+
| ``x_pixel_bounds``     | None                        | A length two list giving the starting  |
|                        |                             | and ending x pixels to use for the     |
|                        |                             | search. `None` uses the image bounds.  |
+------------------------+-----------------------------+----------------------------------------+
| ``x_pixel_buffer``     | None                        | An integer length of pixels outside    |
|                        |                             | the image bounds to use for starting   |
|                        |                             | coordinates. If ``x_bounds`` is        |
|                        |                             | provided that takes precedence.        |
|                        |                             | ``None`` uses the image bounds.        |
+------------------------+-----------------------------+----------------------------------------+
| ``y_pixel_bounds``     | None                        | A length two list giving the starting  |
|                        |                             | and ending y pixels to use for the     |
|                        |                             | search. `None` uses the image bounds.  |
+------------------------+-----------------------------+----------------------------------------+
| ``y_pixel_buffer``     | None                        | An integer length of pixels outside    |
|                        |                             | the image bounds to use for starting   |
|                        |                             | coordinates. If ``y_bounds`` is        |
|                        |                             | provided that takes precedence.        |
|                        |                             | ``None`` uses the image bounds.        |
+------------------------+-----------------------------+----------------------------------------+
