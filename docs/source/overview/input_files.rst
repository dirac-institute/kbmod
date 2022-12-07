Input Files
===========

KBMoD expects Vera C. Rubin Science Pipelines calexp-style FITS files. These are multi-extension fits files that contain:

* photometrically and astometrically calibrated single-CCD image, usually refered to as the "science image",
* variance image, representing per-pixel noise levels, and a
* pixel bitmask

stored in 1st, 2nd and 3rd header extension/plane respectively. The zeroth header extension is expected to contain the image metadata. A collection of science images that overlap the same area on the sky at different times are expected to be grouped into directories, usually refered to as "pointing groups". The path to this directory is a required input to KBMoD, see :ref:`Search Parameters`.

The images are expected to be warped, i.e. geometrically transformed to a set of images with a consistent and uniform relationship between sky coordinates and image pixels on a shared pixel grid. 

Naming scheme
-------------

Each file must include the image’s numeric visit ID in the file name. The parameter `visit_in_filename` (see :ref:`Search Parameters`) indicates character range, not including the directory name, that contains the visit ID in the filename. For example the image file `./my_data/my_data_433932.fits` would use the following parameters::

    im_filepath=“./my_data”
    visit_in_filename=[8, 14]

Time file
---------

There are two cases where you would want to use an external time file:

* when the FITS files do not contain timestamp information
      If no file is included, kbmod will attempt to extract the timestamp from the FITS file header (in the MJD field).
* when you want to prefilter the files based on the parameter `mjd_lims` (see :ref:`Search Parameters`) before loading the file.
      This reduces loading time when accessing a large directory.

The time file provides a mapping of visit ID to timestamp. The time file is an ASCII text file containing two space-separated columns of data: the visit IDs and MJD time of the observation. The first line is a header denoted by `#`::

    # visit_id mean_julian_date
    439116 57162.42540605324
    439120 57162.42863899306
    439124 57162.43279313658
    439128 57162.436995358796
    439707 57163.41836675926
    439711 57163.421717488425



PSF File
--------

The PSF file is an ASCII text file containing two space-separated columns of data: the visit IDs and variance of the PSF for the corresponding observation. The first line is a header denoted by `#`::

    # visit_id psf_val
    439116 1.1
    439120 1.05
    439124 1.4

A PSF file is needed whenever you do not want to use the same default value for every image.


Data Loading
------------

Data is loaded `load_images` function in `analysis_utils.Interface`. This function takes information about the input files (`im_filepath`, `time_file`, `psf_file`, and `visit_in_filename`), bounds on the times to use specified in MJD (`mjd_lims`), and a default PSF (`default_psf`). If then creates an ImageStack object that combines the information from these multiple sources. The ImageStack will include all files in the `im_filepath` that have times within the MJD bounds. Timestamps will be loaded from the input files or the time file. PSFs will be stored with each LayeredImage.

The `load_images` function also returns helper information:
 * img_info - An object containing auxiliary data from the fits files such as their WCS and the location of the observatory.
 * ec_angle - The ecliptic angle for the images as computed using the fits file’s WCS.
