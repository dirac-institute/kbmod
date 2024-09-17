Output Files
============

KBMOD outputs a range of information about the discovered trajectories.

Results Table
-------------

If the ``result_filename`` is provided, KBMOD will serialized the most of the :py:class:`~kbmod.Results` object into a single file. This filename should be the full or relative path and include the ``.ecsv`` suffix.

This results file can be read as::

    results = Results.read_table(filename)

By default the "all_stamps" column is dropped to save space. This can disabled (and one stamp per time step included) by setting the ``save_all_stamps`` configuration parameter to ``True``.

See the notebooks (especially the KBMOD analysis notebook) for examples of how to work with these results.


Legacy Text File
----------------

If the ``legacy_result_filename`` is provided, KBMOD will output the minimal result information (Trajectory details) in a text file format that can be read by numpy.  The main results file includes the found trajectories, their likelihoods, and fluxes.
