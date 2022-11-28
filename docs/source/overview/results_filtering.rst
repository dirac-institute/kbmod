Results analysis
================

The output files contain the set of all trajectories discovered by KBMoD. Many of these trajectories are false positive detections, some area already known objects and, because of the way KBMoD performs the search, some are duplicates. In the following sections we describe the various steps that remove unwanted trajectories from the set of results. 


Filtering
---------

TODO


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
----------------------

Known object matching compares found trajectories against known objects from either SkyBot or the JPL Small Bodies API. Specifically, KBMoD uses the WCS in each FITS file to query the location on the sky that is covered by the image. The found trajectories are then compared against the known objects by checking their relative predicted positions in `(ra, dec)` at each timestep. Objects that are within the threshold for all timesteps are said to match. The number of known objects and matches are displayed.

Known object matching is included for debugging purposes to provide signals into whether there could be known objects in the images and KBMoD’s ability to extract them. All matching is approximate (e.g. KBMoD uses a linear trajectory model) and matching might not be comprehensive. All scientific studies should conduct their own matching analysis.

Relevant matching parameters include:

* `known_obj_thresh` - The matching threshold (in arcseconds) to use. If no threshold is provided (known_obj_thresh = None) then no matching is performed.
* `known_obj_jpl` - Use the JPL API instead of SkyBot.

Acknowledgements
________________

The known object matching uses the IMCCE's SkyBoT VO tool (Berthier et. al. 2006) and JPL’s SSD (Solar System Dynamics) API service.
