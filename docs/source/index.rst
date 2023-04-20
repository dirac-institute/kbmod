.. KBMoD documentation master file, created by
   sphinx-quickstart on Tue Nov 22 22:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/kbmod.svg
  :class: big-logo
  :width: 400
  :alt: KBMOD logo

KBMOD (Kernel Based Moving Object Detection) is a GPU-accelerated framework for the detection of slowly moving asteroids within sequences of images. KBMOD enables users to detect moving objects that are too dim to be detected in a single image without requiring source detection in any individual image, nor at any fixed image cadence. KBMOD achieves this by “shift-and-stacking” images for a range of asteroid velocities and orbits without requiring the input images to be transformed to correct for the asteroid motion.

.. Important:: If you use KBMOD for work presented in a publication or talk please help the project via proper citation or acknowledgment `Whidden et. al. 2019 <https://ui.adsabs.harvard.edu/abs/2019AJ....157..119W/abstract>`_ and the Zenodo link for the code is `here <https://zenodo.org/record/7666852#.Y_ZtauzMKqU>`_. In addition, if you use the known object matching functionality please cite either `IMCCE's SkyBoT VO tool <https://vo.imcce.fr/webservices/skybot/>`_ (Berthier et. al. 2006) or JPL’s SSD (Solar System Dynamics) `API service <https://ssd.jpl.nasa.gov/>`_ as appropriate.

.. grid:: 2

    .. grid-item-card::
        :img-top: _static/getting_started.svg

        Getting Started
	^^^^^^^^^^^^^^^
	
        Just found out about KBMOD? Crash course to what KBMOD can do,
	guided through Jupyter Notebooks. Recommended to anyone looking
	to see KBMOD in action.

        +++

        .. button-ref:: examples/index
            :expand:
            :color: secondary
            :click-parent:

            To the examples

    .. grid-item-card::
        :img-top: _static/user_guide.svg

        User Manual
        ^^^^^^^^^^^

	An in-depth guide through the basic concepts used by KBMOD. Recommended
	to anyone looking to use KBMOD in their work. 

        +++

        .. button-ref:: user_manual/index
            :expand:
            :color: secondary
            :click-parent:

            To the user manual

    .. grid-item-card::
        :img-top: _static/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The API reference guide contains a detailed description of the functions,
        modules, and objects included in KBMOD, which parameters to use and what
	to expect as a returned value. For those interested in contributing, or
	using KBMOD in their own work.

        +++

        .. button-ref:: api_reference/index
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: _static/contributor.svg

        Contributors 
        ^^^^^^^^^^^^

        Want to cite KBMOD? See changelog or release history? Nuts and bolts
	of KBMOD mainly intended for developers and contributors to KBMOD. 

        +++

        .. button-ref:: project_details/index
            :expand:
            :color: secondary
            :click-parent:

            To the developers pages

	
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. rst-class:: hidden

    .. toctree::
       :maxdepth: 1
	
       user_manual/index

    .. toctree::
       :maxdepth: 1
	      
       examples/index
	
	
    .. toctree::
       :maxdepth: 1
		      
       api_reference/index
	
	
    .. toctree::
       :maxdepth: 1
	
       project_details/index

