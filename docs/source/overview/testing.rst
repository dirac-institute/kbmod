Testing
=======

KBMoD provides multiple mechanisms for testing the code and confirming that code changes do not break the code or change the behavior. These include comprehensive unit tests, a regression test, and a diff test.


Unit Tests
----------

KBMoD’s unit tests are included in the `/test` directory and start with the prefix `test_`. The unit test suite can be run using:

```python -m unittest```

from within that directory.


Regression Test
---------------

The regression test generates fake data by creating images with Gaussian noise, inserting fake objects, writing those images to temporary files, calling KBMoD on the temporary files, and comparing the candidate trajectories with the fake objects inserted. The regression test passes if all inserted trajectories are recovered.

You can run the regression test from the `/tests` directory using: ```python regression_test.py```

Additional command line arguments allow you to change the behavior:
 * `num_times` - The number of times (image files) to use.
 * `obs_per_night` - How many observations are taken per night. The regression test generates `num_times` images in clusters of `obs_per_night`. Within in a night observations are separated by 0.01 MJD (~14.4 minutes).
 * `flux` - The flux level to use for the inserted objects.
 * `default_psf` - The default PSF to use.


Diff Test
---------

The diff test is provided to allow users to compare the results of KBMoD runs on real world data sets. This diff test requires that you have already downloaded or created a preprocessed data set (with warped images). The parameter `data_filepath` is used to point to a directory of these files. The diff test operates by running the full KBMoD algorithm and comparing the results of the output files. The diff test passes if the output files are identical within a very small noise threshold (to account for floating point errors).

Before running the diff test on new code, you must first generate golden files that will be used for comparisons. Build a clean version of the code and run the diff test using the `generate_goldens` flag. This will save the run’s output files to a `goldens` directory.

Once the golden files are generated, you can test new code by running the diff test without the `generate_goldens` flag. This will run KBMoD, save the results to a temporary directory, and compare the results and coadded stamp files to the ones in the goldens directory.

The diff test takes a few parameters:
 * `data_filepath` - The location of the data files on which to test the code.
 * `generate_goldens` - Whether to (re)generate the golden files from the current code.
 * `time_filepath` - The path and filename of an external list of timestamps if one is needed.
 * `psf_filepath` - The path and filename of an external list of PSFs if one is needed.
 * `short` - Runs the diff test using a small set of search trajectories to limit the time needed.
 
Depending on the changes you are testing, you might need to change the algorithm’s parameters. These are set in the `input_parameters` dictionary in `diff_test.py`. For example if you want to test a change made to median stamp generation, you would want to set::

    "stamp_type": “median”

Remember to regenerate the goldens from clean code for any major changes in the parameters or the diff test will potentially catch the change in parameters themselves.
