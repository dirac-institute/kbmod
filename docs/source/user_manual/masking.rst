Masking
=======

The KBMOD algorithm uses a data mask to represent invalid pixel values that should be ignored during the search. Masking is applied by:

* applying the per-image mask
* applying a computed a global mask
* growing the computed mask footprint

As stated previously (see :ref:`Input Files`), KBMOD expects  Vera C. Rubin Science Pipelines calexp-style FITS files. Therefore each science image has an associated mask. Per-image masks are applied by checking whether the associated mask image contains one or more of the specified flags, see :py:attr:`~kbmod.run_search.run_search.default_flag_keys` attribute of :py:class:`~kbmod.run_search.run_search`.
These values can be overwritten by providing the :py:attr:`~kbmod.run_search.run_search.flag_keys` attribute  of :py:class:`~kbmod.run_search.run_search`. Futhermore, pixels with values that exceed a threshold value will be masked if a flux threshold is provided, see ``mask_threshold`` key in :ref:`Search Parameters`.

Global mask is computed for all images by counting how many individual masks have flagged a pixel for **any** reason in the list of allowed global masking flags, and masking it out if that count surpasses a given threshold. The number of times a pixel has to be flagged to be masked can be set by ``mask_num_images`` parameter (see :ref:`Search Parameters`) and the list of flags to use in global masking can be modified via the ``repeated_flag_keys`` parameter and is by default set to :py:attr:`~kbmod.run_search.run_search.default_repeated_flag_keys`.
Global masking is not applied when the list of allowed masking flags is empty.

After the per-image and global masks are applied to every image, KBMOD grows the mask to nearby pixels. The parameters ``mask_grow`` (see :ref:`Search Parameters`) determines the ammount of the growth.

The provided pixel bitmask uses the following mapping between flag and bitmask values, which corresponds to the Rubin Science Pipelines mask values:

==================  =====
Key                 Value
==================  =====
BAD                   0
CLIPPED               9
CR                    3
CROSSTALK            10
DETECTED              5 
DETECTED_NEGATIVE     6 
EDGE                  4 
INEXACT_PSF          11 
INTRP                 2 
NOT_DEBLENDED        12 
NO_DATA               8 
REJECTED             13 
SAT                   1 
SENSOR_EDGE          14 
SUSPECT               7 
UNMASKEDNAN          15
==================  =====


To overwrite the default mapping set ``mask_bits_dict`` parameter, see :ref:`Search Parameters`. 
