Results analysis
================

The output files contain the set of all trajectories discovered by KBMOD. Many of these trajectories are false positive detections, some area already known objects and, because of the way KBMOD performs the search, some are duplicates. In the following sections we describe the various steps that remove unwanted trajectories from the set of results. 


Filtering
---------

KBMOD uses two stages of filtering to reduce the number of candidate trajectories. The first stage uses the candidate trajectory's light curve and the second uses the coadded stamp generated from the trajectory's predicted positions.

Clipped SigmaG Filtering
------------------------

During the light curve filtering phase, KBMOD computes the predicted positions at each time steps, assembles a light curve, and looks for statistical outliers along this light curve using clipped-sigmaG filtering. This function identifies outlier points along the likelihood curve and marks them as invalid points. The candidate's overall likelihood is recomputed using only the valid points. The entire candidate trajectory is filtered if less than three valid points remain or the new likelihood is below the threshold defined by the ``lh_level`` parameter. Additional parameters, such as ``sigmaG_lims`` are used to control the light curve filtering.

Relevant light curve filtering parameters include:
 * ``clip_negative`` - Whether to remove all negative values during filtering.
 * ``chunk_size`` - The number of candidate trajectories to filter in a batch. Used to control memory usage.
 * ``gpu_filter`` - Perform an initial round of sigmaG filtering on GPU.
 * ``lh_level`` - The minimum likelihood for a candidate trajectory.
 * ``max_lh`` - The maximum likelihood to keep.
 * ``sigmaG_lims`` - The percentiles for sigmaG filtering (default of [25, 75]).

Stamp Filtering
---------------

The stamp filtering stage is only applied if the ``do_stamp_filter`` parameter is set to True. This stage creates a single stamp representing the sum, mean, or median of pixel values for the stamps at each time step. The stamp type is defined by the ``stamp_type`` parameter and can take on values ``median``, ``mean``, or ``sum``. All of the stamp types drop masked pixels from their computations. The mean and median sums are computed over only the valid time steps from the light curve filtering phase (dropping stamps with outlier fluxes). The sum coadd uses all the time steps regardless of the first phase of filtering.

The stamps are filtered based on how closely the pixel values in the stamp image represent a Gaussian defined with the parameters:
* ``center_thresh`` - The percentage of flux in the central pixel. For example setting this to 0.9 will require that the central pixel of the stamp has 90 percent of all the flux in the stamp. 
* ``peak_offset`` - How far the brightest pixel is from the center of the stamp (in pixels). For example a peak offset of [2.0, 3.0] requires that the brightest pixel in the stamp is at most 2 pixels from the center in the x-dimension and 3-pixels from the center in the y-dimension.
* ``mom_lims`` - Compute the Gaussian moments of the image and compares them to the thresholds.

Relevant stamp filtering parameters include:
 * ``center_thresh`` - The percentage of flux in the central pixel.
 * ``chunk_size`` - The number of candidate trajectories to filter in a batch. Used to control memory usage.
 * ``do_stamp_filter`` - A Boolean indicating whether to generate and filter stamps.
 * ``peak_offset`` - A length 2 list indicating how far the peak is from the center of the stamp in each of the x and y dimensions.
 * ``mom_lims`` -  A length 5 list providing thresholds on the images moments.
 * ``stamp_radius`` - The radius of the stamps to use.

Note that stamps are only generated and output into files if ``do_stamp_filter`` is set to true.

The user can also define custom filters and apply additional filters. For more details see :ref:`Custom Filtering`


Clustering
----------

Clustering is used to combine duplicates found during the initial search. Since each combination of starting pixels and velocity is considered separately, we might see multiple results corresponding to the same true object. For example, if we have an object starting at pixel (10, 15) we might see enough brightness in an adjacent pixel (10, 16) to register a trajectory starting in that location as well.

In the extreme case imagine a bright object centered at (10, 15) and moving at (vx, vy) = (5.0, -5.0). We might find "matches" for this object using trajectories starting at (10, 15), (9, 15), (9, 14), (10, 16), etc. Worse yet, if we use a fine grid of velocities, we might find matches starting at (10, 15) with velocities (5, -5), (4.9, -5), (5.1, -5), etc. The number of combinations explodes and can swamp the users abilities to sort through the results by hand.

But how do we tell which trajectories are "close"? If we only look at the pixel locations at a given time point (event t=0), we might combined two trajectories with very different velocities that happen to pass near the same pixel at that time. Even if this is not likely for real objects, we might merge a real object with a noisy false detection.

The `scikit-learn <https://scikit-learn.org/stable/>`_ ``DBSCAN`` algorithm performs clustering the trajectories. The algorithm can cluster the results based on a combination of position and velocity angle as specified by the parameter ``cluster_type``, which can take on the values of:

* ``all`` or ``pos_vel`` - Use a trajectory's position at the first time stamp (x, y) and velocity (vx, vy) for filtering
* ``position`` or ``start_position`` - Use only trajctory's (x, y) position at the first timestep for clustering.
* ``mid_position`` - Use the predicted position at the median time as coordinates for clustering.

The way DBSCAN computes distances between the trajectories depends on the encoding used. For positional encodings, such as ``position`` and ``mid_position``, the distance is measured directly in pixels. The ``all`` encoding behaves somewhat similarly. However since it combines positions and velocities (or change in pixels per day), they are not actually in the same space.

Relevant clustering parameters include:

* ``cluster_type`` - The types of predicted values to use when determining which trajectories should be clustered together, including position, velocity, and angles  (if ``do_clustering = True``). Must be one of all, position, or mid_position.
* ``do_clustering`` - Cluster the resulting trajectories to remove duplicates.
* ``eps`` - The distance threshold used by DBSCAN.

See Also
________

* `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN>`_

Known Object Matching
---------------------

Known object matching compares found trajectories against known objects from either SkyBot or the JPL Small Bodies API. Specifically, KBMOD uses the WCS in each FITS file to query the location on the sky that is covered by the image. The found trajectories are then compared against the known objects by checking their relative predicted positions in ``(ra, dec)`` at each timestep. Objects that are within the threshold for all timesteps are said to match. The number of known objects and matches are displayed.

Known object matching is included for debugging purposes to provide signals into whether there could be known objects in the images and KBMOD’s ability to extract them. All matching is approximate (e.g. KBMOD uses a linear trajectory model) and matching might not be comprehensive. All scientific studies should conduct their own matching analysis.

Relevant matching parameters include:

* ``known_obj_thresh`` - The matching threshold (in arcseconds) to use. If no threshold is provided (known_obj_thresh = None) then no matching is performed.
* ``known_obj_jpl`` - Use the JPL API instead of SkyBot.

Acknowledgements
----------------

The known object matching uses the `IMCCE's SkyBoT VO tool <https://vo.imcce.fr/webservices/skybot/>`_ (Berthier et. al. 2006) and JPL’s SSD (Solar System Dynamics) `API service <https://ssd.jpl.nasa.gov/>`_. If you use this functionality, please cite the above sources as appropriate.
