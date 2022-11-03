<img src="https://gist.githubusercontent.com/PWhiddy/d42e66a9dd8e4af205a706f388a90ed4/raw/ae5bb87ada12538289852b58ba8e54b564a81584/kbmod.svg?sanitize=true" alt="logo" width="400" height="160"/>

An image processing library for moving object detection implemented with GPUs.  
Based on a Maximum Likelihood detection algorithm for moving astronomical objects.

[![Build Status](https://travis-ci.org/dirac-institute/kbmod.svg?branch=master)](https://travis-ci.org/dirac-institute/kbmod) [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1342297.svg)](https://doi.org/10.5281/zenodo.1342297)



KBMOD is a set of Python tools to search astronomical images for moving
objects based upon method of maximum likelihood detection.

## Updates and Changes

For a list of major changes, including breaking changes to the code, please see the [Major-Changes](https://github.com/dirac-institute/kbmod/wiki/Major-Changes) wiki page.

## Requirements

To build `kbmod` The packages required to build the code are:
* Cuda Toolkit >= 8.0
* CMake >= 3.12

Ensure that the NVIDIA's `nvcc` compiler is available on your system, for example:
```
nvcc --version
```
It is possible that the compiler is installed but not discoverable. In that case add its location to `PATH`. For example, if using `bash`  do `export PATH=/path/to/cuda:$PATH`. The fault location for CUDA Toolkit installation is usually `/usr/local/cuda-XY.Z** where `XY.Z** represent the CUDA Toolkit version that was installed._
If using `bash` add the appropriate command to `~/.bashrc` in order to avoid having to set it before use.

If CUDA Toolkit is not availible on your system follow their [offical installation instructions](https://developer.nvidia.com/cuda-toolkit). The CUDA Toolkit is also availible via Anaconda as `conda install cudatoolkit-dev`.

## Installation

Clone this repository, including all of its submodules:
```
git clone --recursive https://github.com/dirac-institute/kbmod.git
```

Build
```
cd kbmod
pip install .
```

This builds the package and all the dependencies required to test, run `kbmod` and read the results. To use the additional analysis tools available in the `analysis` module it is necessary to install additional dependencies:
```
pip install .[analysis]
```
Note, however, that some of the dependencies in the `analysis` module require packages and supplementary data that are not installed or provided by KBMoD. 

To verify that the installation was successful run the tests:
```
cd tests/
bash run_tests.bash
```

### For Developers

If you want to contribute to the development of KBMoD, it is recommended that you install it in editable mode:
```
pip install -e .
```
Changes you make to the Python source files will then take immediate effect. To recompile the C++ code it's easiest to re-install the package in editable mode again. 

Optionally, it is possible to build just the C++ code via `cmake`. 
```
cmake -B src/kbmod -S .
cmake --build src/kbmod --clean-first
```
To rebuild, it is sufficient to just re-run the `cmake --build` command. Optionally, invoke the cmake generated `Makefile` as `make clean && make` from the `src/kbmod` directory.

## Usage

A short example injecting a simulated object into a stack of images, and then recovering it.

```python

from kbmodpy import kbmod as kb
import numpy as np

# Create a point spread function
psf = kb.psf(1.5)

# load images from list of file paths
imgs =  [ kb.layered_image(file, psf) for file in example_files ]

# Specify an artificial object
flux = 175.0
position = (100.7, 150.3)
velocity = (50, 35)

# Inject object into images
for im in imgs:
    im.add_object(position[0]+im.get_time()*velocity[0], 
                  position[1]+im.get_time()*velocity[1], 
                  flux)

# Recover the object by searching a wide region
velocity_guess = (40, 40)
radius = 20
min_lh = 9.0
min_obs = 10
stack = kb.image_stack(imgs)
search = kb.stack_search(stack)
results = search.region_search(*velocity_guess, radius, min_lh, min_obs)

```

[Short Demonstration](notebooks/Quick_Test.ipynb)

[Processing Real Images](notebooks/HITS_Main_Belt_Comparison.ipynb)

## Reference

[API Reference](notebooks/Kbmod_Reference.ipynb).

## License

The software is open source and available under the BSD license.
