Search Parameters
=================

Search parameters are set extensively via the :py:attr:`~kbmod.run_search.run_search.config` object. There are two methods for setting these parameters. First, you can provide a YAML file of the parameters using the ``config_file`` parameter. Second, you can pass in a dictionary mapping parameter name to parameter value. The dictionary values take precedence over all other settings, allowing you to use KBMOD as part of an internal loop over parameters. 

This document serves to provide a quick overview of the existing parameters and their meaning. For more information refer to the :ref:`User Manual` and :py:class:`~kbmod.run_search.run_search` documentation.

+------------------------+-----------------------------+----------------------------------------+
| **Parameter**          | **Default Value**           | **Interpretation**                     |
+------------------------+-----------------------------+----------------------------------------+
| ``center_thresh``      | 0.00                        | The minimum fraction of total flux     |
|                        |                             | within a stamp that must be contained  |
|                        |                             | in the central pixel                   |
|                        |                             | (if ``do_stamp_filter=True``).         |
+------------------------+-----------------------------+----------------------------------------+
| ``chunk_size``         | 500000                      | The batch size to use when processing  |
|                        |                             | the results of the on-GPU search.      |
+------------------------+-----------------------------+----------------------------------------+
| ``clip_negative``      | False                       | An option used with sigmaG filtering,  |
|                        |                             | remove all negative values prior to    |
|                        |                             | computing the percentiles.             |
+------------------------+-----------------------------+----------------------------------------+
| ``cluster_eps ``       | 20.0                        | The threshold to use for clustering    |
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
|                        |                             | "weighted".
|                        |                             | The filtering coadd is controlled by   |
|                        |                             | the ``stamp_type`` parameter.          |
+------------------------+-----------------------------+----------------------------------------+
| ``debug``              | False                       | Display debugging output.              |
+------------------------+-----------------------------+----------------------------------------+
| ``do_clustering``      | True                        | Cluster the resulting trajectories to  |
|                        |                             | remove duplicates and known objects.   |
|                        |                             | See :ref:`Clustering` for more.        |
+------------------------+-----------------------------+----------------------------------------+
| ``do_mask``            | True                        | Perform masking. See :ref:`Masking`.   |
+------------------------+-----------------------------+----------------------------------------+
| ``do_stamp_filter``    | True                        | Apply post-search filtering on the     |
|                        |                             | image stamps.                          |
+------------------------+-----------------------------+----------------------------------------+
| ``encode_num_bytes``   | -1                          | The number of bytes to use to encode   |
|                        |                             | ``psi`` and ``phi`` images on GPU. By  |
|                        |                             | default a ``float`` encoding is used.  |
|                        |                             | When either ``1`` or ``2``, the images |
|                        |                             | are compressed into ``unsigned int``.  |
+------------------------+-----------------------------+----------------------------------------+
| ``flag_keys``          | default_flag_keys           | Flags used to create the image mask.   |
|                        |                             | See :ref:`Masking`.                    |
+------------------------+-----------------------------+----------------------------------------+
| ``gpu_filter``         | False                       | Perform the filtering on the GPU. Only |
|                        |                             | ``filter_type=clipped_sigmaG``         |
|                        |                             | filtering is supported on GPU.         |
+------------------------+-----------------------------+----------------------------------------+
| ``generator_config``   | None                        | The configuration dictionary for the   |
|                        |                             | trajectory generator that will create  |
|                        |                             | the search candidates.                 |
+------------------------+-----------------------------+----------------------------------------+
| ``im_filepath``        | None                        | The image file path from which to load |
|                        |                             | images. This should point to a         |
|                        |                             | directory with multiple FITS files     |
|                        |                             | (one for each exposure).               |
+------------------------+-----------------------------+----------------------------------------+
| ``legacy_filename` `   | None                        | The full path and file name for the    |
|                        |                             | legacy text file of results. If        |
|                        |                             | ``None`` does not output this file.    |
+------------------------+-----------------------------+----------------------------------------+
| ``lh_level``           | 10.0                        | The minimum computed likelihood for an |
|                        |                             | object to be accepted.                 |
+------------------------+-----------------------------+----------------------------------------+
| ``mask_threshold``     | None                        | The flux threshold over which a pixel  |
|                        |                             | is automatically masked. ``None``      |
|                        |                             | means no flux-based masking.           |
+------------------------+-----------------------------+----------------------------------------+
| ``max_lh``             | 1000.0                      | A maximum likelihood threshold to apply|
|                        |                             | to detected objects. Objects with a    |
|                        |                             | computed likelihood above this         |
|                        |                             | threshold are rejected.                |
+------------------------+-----------------------------+----------------------------------------+
| ``mom_lims``           | [35.5, 35.5, 2.0, 0.3, 0.3] | Thresholds for the moments of a        |
|                        |                             | Gaussian fit to the flux, specified as |
|                        |                             | ``[xx, yy, xy, x, y]``.                |
|                        |                             | If ``do_stamp_filter=True``.           |
+------------------------+-----------------------------+----------------------------------------+
| ``num_obs``            | 10                          | The minimum number of non-masked       |
|                        |                             | observations for the object to be      |
|                        |                             | accepted.                              |
+------------------------+-----------------------------+----------------------------------------+
| ``peak_offset``        | [2.0, 2.0]                  | How far, in pixels, the brightest pixel|
|                        |                             | in the stamp can be from the central   |
|                        |                             | pixel in each direction ``[x,y]``.     |
|                        |                             | If ``do_stamp_filter=True``).          |
+------------------------+-----------------------------+----------------------------------------+
| ``psf_val``            | 1.4                         | The value for the standard deviation of|
|                        |                             | the point spread function (PSF).       |
+------------------------+-----------------------------+----------------------------------------+
| ``repeated_flag_keys`` | default_repeated_flag_keys  | The flags used when creating the global|
|                        |                             | mask. See :ref:`Masking`.              |
+------------------------+-----------------------------+----------------------------------------+
| ``result_filename``    | None                        | Full filename and path for a single    |
|                        |                             | tabular result saves as ecsv.          |
|                        |                             | Can be use used in addition to         |
|                        |                             | outputting individual result files.    |
+------------------------+-----------------------------+----------------------------------------+
| ``results_per_pixel``  | 8                           | The maximum number of results to       |
|                        |                             | to return for each pixel search.       |
+------------------------+-----------------------------+----------------------------------------+
| ``save_all_stamps``    | True                        | Save the individual stamps for each    |
|                        |                             | result and timestep.                   |
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
| ``stamp_type``         | sum                         | The type of coadd to use during stamp  |
|                        |                             | filtering (if ``do_stamp_filter=True``)|
|                        |                             | if:                                    |
|                        |                             | * ``sum`` - (default) Per pixel sum    |
|                        |                             | * ``median`` - Per pixel median        |
|                        |                             | * ``mean`` - Per pixel mean            |
|                        |                             | * ``weighted`` - Per pixel mean        |
|                        |                             |   weighted by 1.0 / variance.          |
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
