.. KBMoD documentation master file, created by
   sphinx-quickstart on Tue Nov 22 22:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/kbmod.svg
  :width: 400
  :alt: KBMOD logo

KBMOD (Kernel Based Moving Object Detection) is a GPU-accelerated framework for the detection of slowly moving asteroids within sequences of images. KBMOD enables users to detect moving objects that are too dim to be detected in a single image without requiring source detection in any individual image, nor at any fixed image cadence. KBMOD achieves this by “shift-and-stacking” images for a range of asteroid velocities and orbits without requiring the input images to be transformed to correct for the asteroid motion.

.. Important:: If you use KBMOD for work presented in a publication or talk please help the project via proper citation or acknowledgement `Whidden et. al. 2019 <https://ui.adsabs.harvard.edu/abs/2019AJ....157..119W/abstract>`_ and the Zenodo link for the code is `here <https://zenodo.org/record/7666852#.Y_ZtauzMKqU>`_. In addition, if you use the known object matching functionality please cite either `IMCCE's SkyBoT VO tool <https://vo.imcce.fr/webservices/skybot/>`_ (Berthier et. al. 2006) or JPL’s SSD (Solar System Dynamics) `API service <https://ssd.jpl.nasa.gov/>`_ as appropriate.

	       
Getting Started
===============

.. toctree::
   :maxdepth: 1

   overview/overview
   overview/input_files
   overview/masking
   overview/search_params
   overview/output_files
   overview/results_filtering
   overview/testing

.. This then should be whatever else we want it to and does not need to be a dry list of all automodule commands

User Documentation
==================

.. toctree::
   :maxdepth: 1
	      
   run_search_referenceapi
   search_referenceapi
   analysis_utils
   image_info
   file_utils
   jointfit_functions
   kbmod_info
   kbmodpy
   result_list
   fake_data_creator


For Developers
==============

.. toctree::
   :maxdepth: 1

   overview/release_notes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
