Output Files
============

KBMOD computes and outputs a range of information about the proposed trajectories that can be used for further analysis.


Results Table
-------------

KBMOD stores all of the result information in a :py:class:`~kbmod.results.Results` object, which provides a wrapper around an AstroPy Table. Most users can treat the produced :py:class:`~kbmod.results.Results` object as a table and access columns directly. However, internally the class provides a range of helper functions to create derived columns described below.

At a minimum, the results table includes the basic trajectory information, including:

* the positions and velocities in pixel space (`x`, `y`, `vx`, and `vy`)
* basic statistics (`likelihood`, `flux`, and `obs_count`).

This information can be access directly with the ``[]`` notation::
    
    lh_0 = results["likelihood"][0]

By default the results table also includes derived information such as the series of psi and phi values from the shift and stack algorithm (``psi_curve`` and ``phi_curve``), a vector of which time steps were marked valid by sigma-G (``obs_valid``), coadded stamps, the corresponding RA, dec in both the search images and (if applicable) the original, un-reprojected images. The time series are all the same length with a single entry for each timestep in the searched data.

A list of all available columns can be obtained with::

    print(results.colnames)

**Coadded Stamps**

The coadded stamp information is controlled by the ``coadds`` and ``stamp_radius`` configuration parameters. The ``coadds`` parameter takes a list of which coadds to include in the results table, including:

* ``mean`` - The mean pixel value (with 'no data' values dropped).
* ``median`` - The median pixel value (with 'no data' values dropped).
* ``sum`` - The sum of pixel values over all times (with no data mapping to 0.0).
* ``weighted`` - The weighted average of pixel values using 1.0 / variance as the weighting function  (with 'no data' values dropped). 

Each coadd is stored in its own column with the name ``coadd_<type>``.  For more information on the stamps, see :ref:`Results Filtering`.

**Individual Stamps**

The set of Individual stamps for each time step can also be stored in the results table by setting the ``save_all_stamps`` configuration parameter to ``True``. This will add a column ``all_stamps`` to the results table, which contains a list of the individual image stamps for each trajectory. Since each stamp is itself a small image, this can significantly increase the size of the results table.

**RA, dec Information**

The mapped RA, dec information consists of up to four columns. The columns ``global_ra`` and ``global_dec`` provide the (RA, dec) in the common WCS frame. If the images have been reprojected, this will be the WCS to which they were reprojected. If there is no global WCS given, these columns will not be present.

The columns ``img_ra`` and ``img_dec`` indicate the positions in the original images. These could be the same or different from the global (RA, dec) even for reprojected images. If the reprojection consists of aligning the images, such as correcting for rotation, the coordinates will be the same. In that case, the RA and dec are not actually changing, just the mappping from RA, dec to pixels. However if the reprojection includes a shift of the viewing location, such as with the barycentric reprojection, we would expect the RA and dec to also change.

**Predicted x, y Information**

KBMOD will also listed the predicted (x, y) pixel coordinates of the object for each time step. The columns ``pred_x`` and ``pred_y`` list the predicted x and y positions in the common WCS frame that KBMOD used for the search.  The columns ``img_x`` and ``img_y`` list the predicted x and y positions in each image's original WCS frame. The ``img_`` columns may be identical to the ``pred_`` columns if the images were not reprojected.

**Metadata**

The table also includes some basic metadata about the set of images, including the number of images (``num_img``), the image dimensions (``dims``), and the midpoint times of the observations (``mid_mjd``). See the :py:class:`~kbmod.results.Results` documentation for more information on what data is saved and how it can be accessed.


Results File
------------

If the ``result_filename`` configuration parameter is provided, KBMOD will serialize most of the :py:class:`~kbmod.results.Results` object into a single file. This filename should be the full or relative path and include the ``.ecsv`` suffix.

This results file can be read as::

    results = Results.read_table(filename)

The code provides configuration options to drop unwanted columns (such as the ``all_stamps`` column described above) or save them to their own files.

The ``drop_columns`` configuration parameter takes a list of column names to be removed from the results table before writing.

The ``separate_col_files`` parameter takes a list of column names and saves that data separately in a file named ``{filename}_{column_name}.fits`` for image columns, such as stamps. For tables it uses the filename ``{filename}_{column_name}.{suffix}`` where the suffix matches the one provided for the results table file itself. These files are also removed from the main table before writing, so they do not occur in that file.


ML Filtering
------------

The results file can be further filtered using a neural network model trained on image stamp data via the `KBMOD ML <https://github.com/dirac-institute/kbmod-ml>`_ package.  See the documentation in that repository for more information.


Config Files
------------

If the ``save_config`` configuration parameter is set to ``True``, KBMOD will save a copy of the configuration file used for the run. The file is added to a (possibly new) directory ``{result_filename}_provenance`` and saved as ``{result_filename}_config.yaml``.