Search Parameters
=================

Search parameters are set extensively via the :py:attr:`kbmod.run_search.run_search.config` dictionary. This document serves to provide a quick overview of the existing parameters and their meaning. For more information refer to the :ref:`Overview` and :py:class:`~kbmod.run_search.run_search` documentation.

==================   ===========================  ===========================================================================================================================================
Parameter            Default Value                Interpretation
==================   ===========================  ===========================================================================================================================================
im_filepath          None                         The image file path from which to load images. This should point to a directory with multiple FITS files (one for each exposure).
res_filepath         None                      	  -
time_file            None                      	  The path and filename of a separate file containing the time when each image was taken. See :ref:`Time File` for more.
psf_file             None                         The path and filename of a separate file containing the per-image PSFs. See :ref:`PSF File` Format for more.
v_arr                [92.0, 526.0, 256]       	  Minimum, maximum and number of velocities to search through.
ang_arr              [np.pi/15, np.pi/15, 128] 	  Minimum, maximum and number of angles to search through.
output_suffix        search                       Suffix appended to output filenames. See :ref:`Output Files` for more.
mjd_lims             None                         Limits the search to images taken within the given range (or None for no filtering).
average_angle        None                         Overrides the ecliptic angle calculation and instead centers the average search around average_angle.
do_mask              True                         Perform masking. See :ref:`Masking` for more.
mask_num_images      2				  Threshold for number of times a pixel needs to be flagged in order to be masked in global mask. See :ref:`Masking` for more.
mask_threshold       None			  The flux threshold over which a pixel is automatically masked. `None` means no flux-based masking.
mask_grow            10				  Size, in pixels, the mask will be grown by. 
lh_level             10.0			  -
psf_val              1.4			  -
num_obs              10				  -
num_cores            1				  -
visit_in_filename    [0, 6]			  Character range that contains the visit ID. See :ref:`Naming Scheme` for more.
sigmaG_lims          [25, 75]			  -
chunk_size           500000			  -
max_lh               1000.0			  -
filter_type          clipped_sigmaG		  -
center_thresh        0.00			  -
peak_offset          [2.0, 2.0]			  -
mom_lims             [35.5, 35.5, 2.0, 0.3, 0.3]  -
stamp_type           sum			  -
stamp_radius         10				  -
eps                  0.03			  -
gpu_filter           False			  -
do_clustering        True			  -
do_stamp_filter      True			  -
clip_negative        False			  -
sigmaG_filter_type   lh				  -
cluster_type         all			  -
cluster_function     DBSCAN			  -
mask_bits_dict       default_mask_bits_dict	  -
flag_keys            default_flag_keys		  -
repeated_flag_keys   default_repeated_flag_keys	  -
bary_dist            None			  -
encode_psi_bytes     -1				  -
encode_phi_bytes     -1				  -
known_obj_thresh     None			  -
known_obj_jpl        False			  -
==================   ===========================  ===========================================================================================================================================
