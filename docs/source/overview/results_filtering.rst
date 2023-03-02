Results analysis
================

The output files contain the set of all trajectories discovered by KBMOD. Many of these trajectories are false positive detections, some area already known objects and, because of the way KBMOD performs the search, some are duplicates. In the following sections we describe the various steps that remove unwanted trajectories from the set of results. 


Filtering
---------

KBMOD uses two stages of filtering to reduce the number of candidate trajectories. The first stage uses the candidate trajectory's light curve and the second uses the coadded stamp generated from the trajectory's predicted positions.

During the light curve filtering phase, KBMOD computes the predicted positions at each time steps, assembles a light curve, and looks for statistical outliers along this light curve using clipped-sigmaG filtering. This function identifies outlier points along the likelihood curve and marks them as invalid points. The candidate's overall likelihood is recomputed using only the valid points. The entire candidate trajectory is filtered if less than three valid points remain or the new likelihood is below the threshold defined by the `lh_level` parameter. Additional parameters, such as `sigmaG_lims` are used to control the light curve filtering.

Relevant light curve filtering parameters include:
 * `clip_negative` - Whether to remove all negative values during filtering.
 * `chunk_size` - The number of candidate trajectories to filter in a batch. Used to control memory usage.
 * `gpu_filter` - Perform an initial round of sigmaG filtering on GPU.
 * `lh_level` - The minimum likelihood for a candidate trajectory.
 * `max_lh` - The maximum likelihood to keep.
 * `sigmaG_lims` - The percentiles for sigmaG filtering (default of [25, 75]).

The stamp filtering stage is only applied if the `do_stamp_filter` parameter is set to True. This stage by creating a single stamp representing the sum, mean, or median of pixel values for the stamps at each time step. The stamp type is defined by the `stamp_type` parameter and can take on values `median`, `cpp_median`, `cpp_mean`, `parallel_sum`, or `sum`. All of the stamp types drop masked pixels from their computations. The mean and median sums are computed over only the valid time steps from the light curve filtering phase (dropping stamps with outlier fluxes). The sum coadd uses all the time steps regardless of the first phase of filtering.

The stamps are filtered based on how closely the pixel values in the stamp image represent a guassian (defined with the parameters `center_thresh` (the percentage of flux in the central pixel), `peak_offset` (how far the peak is from the center of the stamp), and `mom_lims` (thresholds on the images moments)). Trajectories with stamps satisfying these thresholds are retained.

Relevant stamp filtering parameters include:
 * `center_thresh` - The percentage of flux in the central pixel.
 * `chunk_size` - The number of candidate trajectories to filter in a batch. Used to control memory usage.
 * `do_stamp_filter` - A Boolean indicating whether to generate and filter stamps.
 * `peak_offset` - A length 2 list indicating how far the peak is from the center of the stamp in each of the x and y dimensions.
 * `mom_lims` -  A length 5 list providing thresholds on the images moments.
 * `stamp_radius` - The radius of the stamps to use.

Note that stamps are only generated and output into files if `do_stamp_filter` is set to true.


Clustering
----------

Clustering is used to combine duplicates found during the initial search. Since each combination of starting pixels and velocity is considered separately, we might see multiple results corresponding to the same true object. For example, if we have an object starting at pixel (10, 15) we might see enough brightness in an adjacent pixel (10, 16) to register a trajectory starting in that location as well.

Two `scikit-learn <https://scikit-learn.org/stable/>`_ algorithms are supported for clustering the trajectories, `DBSCAN` and `OPTICS`, specified by the parameter `cluster_function`. Either algorithm can cluster the results based on a combination of position, velocity, and angle as specified by the parameter cluster_type, which can take on the values of:

* `all` - Use scaled x position, scaled y position, scale velocity, and scaled angle as coordinates for clustering.
* `position` - Use only scaled x position and scaled y position as coordinates for clustering.
* `mid_position` - Use the (scaled) predicted position at the middle time as coordinates for clustering.

Relevant clustering parameters include:

* `cluster_function` - The name of the clustering algorithm used (if `do_clustering = True`). The value must be one of `DBSCAN` or `OPTICS`.
* `cluster_type` - The types of predicted values to use when determining which trajectories should be clustered together, including position, velocity, and angles  (if `do_clustering = True`). Must be one of all, position, or mid_position.
* `do_clustering` - Cluster the resulting trajectories to remove duplicates.

See Also
________

* `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN>`_
* `OPTICS <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html?highlight=optics#sklearn.cluster.OPTICS>`_


Known Object Matching
---------------------

Known object matching compares found trajectories against known objects from either SkyBot or the JPL Small Bodies API. Specifically, KBMOD uses the WCS in each FITS file to query the location on the sky that is covered by the image. The found trajectories are then compared against the known objects by checking their relative predicted positions in `(ra, dec)` at each timestep. Objects that are within the threshold for all timesteps are said to match. The number of known objects and matches are displayed.

Known object matching is included for debugging purposes to provide signals into whether there could be known objects in the images and KBMOD’s ability to extract them. All matching is approximate (e.g. KBMOD uses a linear trajectory model) and matching might not be comprehensive. All scientific studies should conduct their own matching analysis.

Relevant matching parameters include:

* `known_obj_thresh` - The matching threshold (in arcseconds) to use. If no threshold is provided (known_obj_thresh = None) then no matching is performed.
* `known_obj_jpl` - Use the JPL API instead of SkyBot.

Acknowledgements
----------------

The known object matching uses the `IMCCE's SkyBoT VO tool <https://vo.imcce.fr/webservices/skybot/>`_ (Berthier et. al. 2006) and JPL’s SSD (Solar System Dynamics) `API service <https://ssd.jpl.nasa.gov/>`_. If you use this functionality, please cite the above sources as appropriate.