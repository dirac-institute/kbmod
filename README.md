<img src="https://gist.githubusercontent.com/PWhiddy/d42e66a9dd8e4af205a706f388a90ed4/raw/ae5bb87ada12538289852b58ba8e54b564a81584/kbmod.svg?sanitize=true" alt="logo" width="400" height="160"/>

An image processing library for moving object detection implemented with GPUs.  
Based on a Maximum Likelihood detection algorithm for moving astronomical objects.

[![Build Status](https://github.com/dirac-institute/kbmod/actions/workflows/test_build.yaml/badge.svg)](https://github.com/dirac-institute/kbmod/actions/workflows/test_build.yaml)[![Documentation](https://github.com/dirac-institute/kbmod/actions/workflows/build_docs.yaml/badge.svg)](https://epyc.astro.washington.edu/~kbmod/) [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1342297.svg)](https://doi.org/10.5281/zenodo.1342297)



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
It is possible that the compiler is installed but not discoverable. In that case add its location to `PATH`. For example, if using `bash`  do `export PATH=/path/to/cuda:$PATH`. The default location for CUDA Toolkit installation is usually `/usr/local/cuda-XY.Z**` where `XY.Z` represent the CUDA Toolkit version that was installed.    
If using `bash` add the appropriate command to `~/.bashrc` in order to avoid having to set it repeatedly.

If CUDA Toolkit is not availible on your system follow their [offical installation instructions](https://developer.nvidia.com/cuda-toolkit). Optionally, if you use Anaconda virtual environments, the CUDA Toolkit is also availible as `conda install cudatoolkit-dev`.

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

This builds the package and all the dependencies required to test, run KBMoD on images and read the results. To use the additional analysis tools available in the `analysis` module it is necessary to install additional dependencies:
```
pip install .[analysis]
```
Note, however, that some of the dependencies in the `analysis` module require packages and supplementary data that are not installed nor provided by KBMoD. 

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

It is possible to build only the C++ code via `cmake`. 
```
cmake -B src/kbmod -S .
cmake --build src/kbmod --clean-first
```
To rebuild, it is sufficient to just re-run the `cmake --build` command. Optionally, invoke the cmake generated `Makefile` as `make clean && make` from the `src/kbmod` directory.

## Usage

A short example injecting a simulated object into a stack of images, and then recovering it.

```python

import kbmod.search as kb
import numpy as np

# Create a point spread function
psf = kb.psf(1.5)

# Create fake data with ten 512x512 pixel images.
from kbmod.fake_data_creator import *
ds = FakeDataSet(512, 512, 10)
imgs = ds.stack.get_images()

# Alternatively, if you have real images you would like to use,
# load them from files as layered_images:
# imgs =  [ kb.layered_image(file, psf) for file in example_files ]

# Get the timestamp of the first image.
t0 = imgs[0].get_time()
print(f"Image times start at {t0}.")

# Specify an artificial object
flux = 275.0
position = (10.7, 15.3)
velocity = (2, 0)

# Inject object into images
for im in imgs:
    dt = im.get_time() - t0
    im.add_object(position[0] + dt * velocity[0], 
                  position[1] + dt * velocity[1], 
                  flux)

# Create a new image stack with the inserted object.
stack = kb.image_stack(imgs)

# Recover the object by searching a set of trajectories.
search = kb.stack_search(stack)
search.search(
    5,  # Number of search angles to try (-0.1, -0.05, 0.0, 0.05, 0.1)
    5,  # Number of search velocities to try (0, 1, 2, 3, 4)
    -0.1,  # The minimum search angle to test
    0.1,  # The maximum search angle to test
    0,  # The minimum search velocity to test
    4,  # The maximum search velocity to test
    7,  # The minimum number of observations
)

# Get the top 10 results.
results = search.get_results(0, 10)
print(results)
```

## Reference

[API Reference](notebooks/Kbmod_Reference.ipynb).

## License

The software is open source and available under the BSD license.
