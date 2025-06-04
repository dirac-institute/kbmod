User Manual
===========


.. toctree::
   :maxdepth: 1

   overview
   input_files
   search_space
   reprojection
   search_params
   output_files
   results_filtering
   custom_filtering


Overview
========

For an introduction to KBMOD and its components, see the :ref:`KBMOD Overview` page.

Installation
============

KBMOD can be installed from source and requires:

* Python >= 3.10
* CMake >= 3.23
* A C++ compiler
* Cuda Toolkit >= 8.0 (to build the GPU libraries)

Note that while it is possible to install KBMOD on systems without the CUDA toolkits, the resulting installation will not be able to perform the core search on the GPU and thus will not be as performant.

We recommend using a virtual environment such as conda to install the necessary dependencies:

.. code-block:: bash
 
    conda create -n kbmod_env -c conda-forge python=3.11 cmake>=3.23
    conda activate kbmod_env

Depending on your system, you may need to install additional dependencies. See the Troubleshooting section below for more details.

Once you have the prerequisites installed, you can install KBMOD using a combination of ``git`` and ``pip``.

.. code-block:: bash
		
   git clone https://github.com/dirac-institute/kbmod.git --recursive
   cd kbmod
   pip install .

The ``pip install`` command will install the necessary python dependencies and compile the C++ code.

You can run then run the tests to check that everything works:

.. code-block:: bash
		
   cd tests
   python -m unittest


Running KBMOD
=============
   
To run a search, KBMOD must be provided with

* appropriately pre-processed input data (see :ref:`Input Files`)
* appropriate search and filter parameters (see :ref:`Search Parameters`)

For an introduction to the KBMOD search algorithm, see the :ref:`Search Algorithm and Search Space` page.

The search is initiated via the :py:class:`~~kbmod.run_search.run_search` class and consists of several phases:

* Data is loaded from the input files as specified above (see :ref:`Input Files` for more details).
* The shift and stack approach is used to search for potential trajectories originating from each pixel in the first image.
* The list of potential trajectories is filtered using various metrics (see :ref:`Results Filtering`).
* Remaining trajectories are clustered to remove duplicates. Only one trajectory per cluster is kept (see :ref:`Results Filtering`).
* The found trajectories are output to result files for later analysis (see :ref:`Output Files`).


Troubleshooting
===============

When combining Python, C++, CUDA libraries, and GPU drivers, it is possible that you will run into some environmental complexities. Below we discuss some common debugging techniques and solutions.


Checking Your GPU 
-----------------

You can check that you have a compatible NVIDIA GPU and the necessary drivers installed:

.. code-block:: bash
		
    nvidia-smi

While a GPU is not required to compile the code, it is needed to run any of the algorithms on GPU.


Checking CUDA LIbraries
-----------------------

The ``nvcc`` compiler is part of the CUDA toolkit and is required to compile the GPU libraries. You can check that NVIDIA's ``nvcc`` compiler is available on your system and is visible by running:

.. code-block:: bash
		
   nvcc --version

It is possible that the ``nvcc`` compiler is installed but not discoverable. In that case, you will need to add its location to the system environment ``PATH`` variable by, if using ``bash`` for example, setting:

.. code-block:: bash
		
   export PATH=/path/to/cuda:$PATH

If the ``nvcc`` compiler is not present on the system, you need to install it. If you are using a conda virtual environment, you can install the CUDA Toolkit using conda:

.. code-block:: bash
		
   conda install -c conda-forge cudatoolkit-dev

Alternatively, you can follow NVIDIA's `offical installation instructions <https://developer.nvidia.com/cuda-toolkit>`_.

Note that you will need to make sure the **combination** of the CUDA toolkit version, the C++ compiler, and the GPU drivers are mutually compatible.  We recommend starting by determining the driver version for the GPU that you are using, which can be found with the ``nvidia-smi`` command.  We have found the table in following `github gist <https://gist.github.com/ax3l/9489132>`_ helpful for determining the compiler version needed. In some cases we have needed to force a new compiler to be installed, such as:

.. code-block:: bash
    conda install -c conda-forge gxx=12.2 gcc=12.2 sysroot_linux-64 

Of course, you will want to substitute in the version numbers that are compatible with your specific GPU drivers.  We **highly** recommend that you are using a virtual environment before you start changing compiler versions.


Debugging Build Failures
------------------------

If KBMOD installation fails during ``pip install`` it could be due to problems building the C++ portion of the code (and specifically linking in libraries). By default pip install will suppress most of the output of the C++ building process. You can get more verbose output for debugging by manually building the C++ libraries:

.. code-block:: bash

   cmake -B src/kbmod -S .
   cmake --build src/kbmod --clean-first

The first command will output information about the compiler and libraries that are found on the system. For example, you might see lines like "Looking for a CUDA compiler - NOTFOUND".


Checking the KBMOD Installation
-------------------------------

After installing KBMOD, you can check the version built, whether it built with OpenMP, and whether it can detect a GPU by running: 

.. code-block:: bash
		
   kbmod-version

If the program shows "False" for "GPU Code Enabled" then you are either running it on a system without a compatible GPU or the installation was not able to build and link the CUDA code. You can check the GPU using the ``nvidia-smi`` command as described above.
