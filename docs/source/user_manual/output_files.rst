Output Files
============

KBMOD outputs a range of information about the discovered trajectories.


Results Table
-------------

KBMOD stores all of the result information in a :py:class:`~kbmod.Results` object, which provides a wrapper around an AstroPy Table. Most users can treat the produced :py:class:`~kbmod.Results` object as a table and access columns directly. However, internally the object provides a range of helper functions to create derived columns described below.

At a minimum, the results table includes the basic trajectory information, including:

* the positions and velocities in pixel space (`x`, `y`, `vx`, and `vy`)
* basic statistics (`likelihood`, `flux`, and `obs_count`).

By default it includes additional derived information such as the series of psi and phi values from the shift and stack algorithm (`psi_curve` and `phi_curve`), a vector of which time steps were marked valid by sigma-G (`obs_valid`), coadded stamps, the corresponding RA, dec in both the search images and (if applicable) the original, un-reprojected images.

The coadded stamp information is controlled by the ``coadds`` and ``stamp_radius`` configuration parameters. The ``coadds`` parameter takes a list of which coadds to include in the results table, including:

* ``mean`` - The mean pixel value.
* ``median`` - The median pixel value.
* ``sum`` - The sum of pixel values over all times (with no data mapping to 0.0).
* ``weighted`` - The weighted average of pixel values using 1.0 / variance as the weighting function.

Each coadd is stored in its own column with the name `coadd_<type>`.  For more information on the stamps, see :ref:`Results Filtering`.

The mapped RA, dec information consists of up to four columns. The columns `global_ra` and `global_dec` provide the (RA, dec) in the common WCS frame. If the images have been reprojected, this will be the WCS to which they were reprojected. If there is no global WCS given, these columns will not be present.

The columns `img_ra` and `img_dec` indicate the positions in the original images. These could be the same or different from the global (RA, dec) even for reprojected images. If the reprojection consists of aligning the images, such as correcting for rotation, the coordinates will be the same. In that case, the RA and dec are not actually changing, just the mappping from RA, dec to pixels. However if the reprojection includes a shift of the viewing location, such as with the barycentric reprojection, we would expect the RA and dec to also change.


Results File
------------

If the ``result_filename`` is provided, KBMOD will serialize most of the :py:class:`~kbmod.Results` object into a single file. This filename should be the full or relative path and include the ``.ecsv`` suffix.

This results file can be read as::

    results = Results.read_table(filename)

By default the "all_stamps" column is dropped to save space. This can disabled (and one stamp per time step included) by setting the ``save_all_stamps`` configuration parameter to ``True``.

See the notebooks (especially the KBMOD analysis notebook) for examples of how to work with these results.


ML Filtering
------------

The results file can be further filtered using a neural network model trained on image stamp data via the `KBMOD ML <https://github.com/dirac-institute/kbmod-ml>`_ package.  See the documentation in that repository for more information.
