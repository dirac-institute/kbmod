Input Files
===========


KBMOD expects Vera C. Rubin Science Pipelines calexp-style data. These can be provided as a set of multi-extension FITS files, references to the data's location on a Butler instance, or a saved :py:class:`~kbmod.work_unit.WorkUnit`.

Butler
------

TODO


WorkUnit
--------

The :py:class:`~kbmod.work_unit.WorkUnit` objects provide functions for writing to and loading from files. In addition to image data, the :py:class:`~kbmod.work_unit.WorkUnit` includes configuration data for the run and all necessary metadata (e.g. the WCS). To load a :py:class:`~kbmod.work_unit.WorkUnit` from a file, use::

    my_wu = WorkUnit.from_fits(input_filename)


FITS Files
----------

If loading data from raw FITS files, these must be Vera C. Rubin Science Pipelines calexp-style FITS files that contain:

* photometrically and astometrically calibrated single-CCD image, usually referred to as the "science image",
* variance image, representing per-pixel noise levels, and a
* pixel bitmask

stored in 1st, 2nd and 3rd header extension/plane respectively. The zeroth header extension is expected to contain the image metadata. A single FITS file can be loaded with the :py:meth:`~kbmod.util_functions.load_deccam_layered_image` function, which takes the file name and an optional numpy array of the PSF kernel and produces a :py:class:`~kbmod.core.image_stack_py.LayeredImagePy`.

To build an :py:class:`~kbmod.image_collection.ImageCollection` from multiple FITS files, use the class's :py:meth:`~kbmod.image_collection.ImageCollection.fromDir()` function. The images within a single run are expected to be warped, i.e. geometrically transformed to a set of images with a consistent and uniform relationship between sky coordinates and image pixels on a shared pixel grid. 
