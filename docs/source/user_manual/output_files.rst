Output Files
============

KBMOD outputs a range of information about the discovered trajectories. Depending on the search configuration parameters this data can be output as a single combined file and/or individual files.

Results Table
-------------

If the ``result_filename`` is provided, KBMOD will serialized the most of the :py:class:`~kbmod.Results` object into a single file. This filename should be the full or relative path and include the ``.ecsv`` suffix.

This results file can be read as::

    results = Results.read_table(filename)

By default the "all_stamps" column is dropped to save space. This can disabled (and one stamp per time step included) by setting the ``save_all_stamps`` configuration parameter to ``True``.

See the notebooks (especially the KBMOD analysis notebook) for examples of how to work with these results.


Individual Files
----------------

If the ``res_filepath`` configuration option is provided and ``ind_output_files`` configuration option is set to ``True``, the code will produce a few individual output files are useful on their own. Each filename includes a user defined suffix, allowing user to easily save and compare files from different runs. Below we use SUFFIX to indicate the user-defined suffix.

The main file that most users will want to access is ``results_SUFFIX.txt``. This file contains one line for each trajectory with the trajectory information (x pixel start, y pixel start, x velocity, y velocity), the number of observations seen, the estimated flux, and the estimated likelihood.

The full list of output files is:

* ``all_stamps_SUFFIX.npy`` - All of the postage stamp images for each found trajectory. This is a size ``N`` x ``T`` numpy array where ``N`` is the number of results and ``T`` is the number of time steps.
* ``config_SUFFIX.yml`` - A full dump of the configuration parameters in YAML format.
* ``filter_stats_SUFFIX.csv`` - A CSV mapping each filtering label to the number of results removed at that stage.
* ``results_SUFFIX.txt`` - The main results file including the found trajectories, their likelihoods, and fluxes.
