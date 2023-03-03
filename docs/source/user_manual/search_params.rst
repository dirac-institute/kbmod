Search Parameters
=================

Search parameters are set extensively via the :py:attr:`~kbmod.run_search.run_search.config` dictionary. This document serves to provide a quick overview of the existing parameters and their meaning. For more information refer to the :ref:`User Manual` and :py:class:`~kbmod.run_search.run_search` documentation.

+------------------------+-----------------------------+----------------------------------------+
| **Parameter**          | **Default Value**           | **Interpretation**                     |
+------------------------+-----------------------------+----------------------------------------+
| ``ang_arr``            | [np.pi/15, np.pi/15, 128]   | Minimum, maximum and number of angles  |
|                        |                             | to search through.                     |
+------------------------+-----------------------------+----------------------------------------+
| ``average_angle``      | None                        | Overrides the ecliptic angle           |
|                        |                             | calculation and instead centers the    |
|                        |                             | average search around average_angle.   |
+------------------------+-----------------------------+----------------------------------------+
| ``bary_dist``          | None                        | The barycentric distance to use when   |
|                        |                             | correcting the predicted positions.    |
|                        |                             | If set to None, KBMOD will not use     |
|                        |                             | barycentric corrections.               |
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
| ``cluster_function``   | DBSCAN                      | The name of the clustering algorithm   |
|                        |                             | used (if ``do_clustering=True``). The  |
|                        |                             | value must be one of ``DBSCAN`` or     |
|                        |                             | ``OPTICS``.                            |
+------------------------+-----------------------------+----------------------------------------+
| ``cluster_type``       | all                         | Types of predicted values to use when  |
|                        |                             | determining trajectories to clustered  |
|                        |                             | together, including position, velocity,|
|                        |                             | and angles  (if do_clustering = True). |
|                        |                             | Must be one of ``all``, ``position``,  |
|                        |                             | or ``mid_position``.                   |
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
| ``eps``                | 0.03                        | The epsilon value to use in DBSCAN     |
|                        |                             | clustering (if ``cluster_type=DBSCAN`` |
|                        |                             | and ``do_clustering=True``).           |
+------------------------+-----------------------------+----------------------------------------+
| ``encode_psi_bytes``   | -1                          | The number of bytes to use to encode   |
|                        |                             | ``psi`` images on GPU. By default a    |
|                        |                             | ``float`` encoding is used. When either|
|                        |                             | ``1`` or ``2``, the images are         |
|                        |                             | compressed into ``unsigned int``.      |
+------------------------+-----------------------------+----------------------------------------+
| ``encode_phi_bytes``   | -1                          | The number of bytes to use to encode   |
|                        |                             | ``psi`` images on GPU. By default a    |
|                        |                             | ``float`` encoding is used. When either|
|                        |                             | ``1`` or ``2``, the images are         |
|                        |                             | compressed into ``unsigned int``.      |
+------------------------+-----------------------------+----------------------------------------+
| ``flag_keys``          | default_flag_keys           | Flags used to create the image mask.   |
|                        |                             | See :ref:`Masking`.                    |
+------------------------+-----------------------------+----------------------------------------+
| ``gpu_filter``         | False                       | Perform the filtering on the GPU. Only |
|                        |                             | ``filter_type=clipped_sigmaG``         |
|                        |                             | filtering is supported on GPU.         |
+------------------------+-----------------------------+----------------------------------------+
| ``im_filepath``        | None                        | The image file path from which to load |
|                        |                             | images. This should point to a         |
|                        |                             | directory with multiple FITS files     |
|                        |                             | (one for each exposure).               |
+------------------------+-----------------------------+----------------------------------------+
| ``known_obj_jpl``      | False                       | Use JPL’s API (over ``SkyBot``) to     |
|                        |                             | look up known objects                  |
|                        |                             | (if ``known_obj_thresh!=None``).       |
+------------------------+-----------------------------+----------------------------------------+
| ``known_obj_thresh``   | None                        | The threshold, in arcseconds, used to  |
|                        |                             | compare results to known objects from  |
|                        |                             | JPL or SkyBot.                         |
+------------------------+-----------------------------+----------------------------------------+
| ``lh_level``           | 10.0                        | The minimum computed likelihood for an |
|                        |                             | object to be accepted.                 |
+------------------------+-----------------------------+----------------------------------------+
| ``peak_offset``        | [2.0, 2.0]                  | How far, in pixels, the brightest pixel|
|                        |                             | in the stamp can be from the central   |
|                        |                             | pixel in each direction ``[x,y]``.     |
|                        |                             | If ``do_stamp_filter=True``).          |
+------------------------+-----------------------------+----------------------------------------+
| ``psf_val``            | 1.4                         | The value for the standard deviation of|
|                        |                             | the point spread function (PSF).       |
+------------------------+-----------------------------+----------------------------------------+
| ``mask_bits_dict``     | default_mask_bits_dict      | A dictionary indicating which masked   |
|                        |                             | values to consider invalid pixels.     |
+------------------------+-----------------------------+----------------------------------------+
| ``mask_grow``          | 10                          | Size, in pixels, the mask will be grown|
|                        |                             | by.                                    |
+------------------------+-----------------------------+----------------------------------------+
| ``mask_num_images``    | 2                           | Threshold for number of times a pixel  |
|                        |                             | needs to be flagged in order to be     |
|                        |                             | masked in global mask.                 |
|                        |                             | See :ref:`Masking` for more.           |
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
| ``mjd_lims``           | None                        | Limits the search to images taken      |
|                        |                             | within the given range (or ``None``    |
|                        |                             | for no filtering).                     |
+------------------------+-----------------------------+----------------------------------------+
| ``mom_lims``           | [35.5, 35.5, 2.0, 0.3, 0.3] | Thresholds for the moments of a        |
|                        |                             | Gaussian fit to the flux, specified as |
|                        |                             | ``[xx, yy, xy, x, y]``.                |
|                        |                             | If ``do_stamp_filter=True``.           |
+------------------------+-----------------------------+----------------------------------------+
| ``num_cores``          | 1                           | The number of threads  to use for      |
|                        |                             | parallel filtering.                    |
+------------------------+-----------------------------+----------------------------------------+
| ``num_obs``            | 10                          | The minimum number of non-masked       |
|                        |                             | observations for the object to be      |
|                        |                             | accepted.                              |
+------------------------+-----------------------------+----------------------------------------+
| ``output_suffix``      | search                      | Suffix appended to output filenames.   |
|                        |                             | See :ref:`Output Files` for more.      |
+------------------------+-----------------------------+----------------------------------------+
| ``repeated_flag_keys`` | default_repeated_flag_keys  | The flags used when creating the global|
|                        |                             | mask. See :ref:`Masking`.              |
+------------------------+-----------------------------+----------------------------------------+
| ``res_filepath``       | None                        | The path of the directory in which to  |
|                        |                             | store the results files.               |
+------------------------+-----------------------------+----------------------------------------+
| ``sigmaG_lims``        | [25, 75]                    | The percentiles to use in sigmaG       |
|                        |                             | filtering, if                          |
|                        |                             | ``filter_type= clipped_sigmaG``.       |
+------------------------+-----------------------------+----------------------------------------+
| ``stamp_radius``       | 10                          | Half the size of a side of a box cut   |
|                        |                             | around the predicted position when     |
|                        |                             | creating a stamp for stamp filtering.  |
+------------------------+-----------------------------+----------------------------------------+
| ``stamp_type``         | sum                         | The type of stamp to use during stamp  |
|                        |                             | filtering (if ``do_stamp_filter=True``)|
|                        |                             | if:                                    |
|                        |                             | * ``sum`` - (default) A simple sum of  |
|                        |                             | all individual stamps                  |
|                        |                             | * ``parallel_sum`` - A faster simple   |
|                        |                             | sum implemented in c++.                |
|                        |                             | * ``cpp_median`` - A faster per-pixel  |
|                        |                             | median implemented in c++              |
|                        |                             | * ``cpp_mean`` - A per pixel mean      |
|                        |                             | implemented in c++.                    |
+------------------------+-----------------------------+----------------------------------------+
| ``time_file``          | None                        | The path and filename of a separate    |
|                        |                             | file containing the time when each     |
|                        |                             | image was taken. See :ref:`Time File`  |
|                        |                             | for more.                              |
+------------------------+-----------------------------+----------------------------------------+
| ``v_arr``              | [92.0, 526.0, 256]          | Minimum, maximum and number of         |
|                        |                             | velocities to search through.          |
+------------------------+-----------------------------+----------------------------------------+
| ``x_pixel_bounds``     | None                        | A length two list giving the starting  |
|                        |                             | and ending x  pixels to use for the    |
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
| ``psf_file``           | None                        | The path and filename of a separate    |
|                        |                             | file containing the per-image PSFs.    |
|                        |                             | See :ref:`PSF File` for more.          |
+------------------------+-----------------------------+----------------------------------------+
| ``visit_in_filename``  | [0, 6]                      | Character range that contains the visit|
|                        |                             | ID. See :ref:`Naming Scheme` for more. |
+------------------------+-----------------------------+----------------------------------------+
