<img src="https://gist.githubusercontent.com/PWhiddy/d42e66a9dd8e4af205a706f388a90ed4/raw/ae5bb87ada12538289852b58ba8e54b564a81584/kbmod.svg?sanitize=true" alt="logo" width="400" height="160"/>

An image processing library for moving object detection implemented with GPUs.  
Based on a Maximum Likelihood detection algorithm for moving astronomical objects.

[![Build Status](https://github.com/dirac-institute/kbmod/actions/workflows/canary_builds.yaml/badge.svg)](https://github.com/dirac-institute/kbmod/actions/workflows/test_build.yaml)[![Documentation](https://github.com/dirac-institute/kbmod/actions/workflows/build_docs.yaml/badge.svg)](https://epyc.astro.washington.edu/~kbmod/) [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1342297.svg)](https://doi.org/10.5281/zenodo.1342297)[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)



KBMOD is a set of Python tools to search astronomical images for moving
objects based upon method of maximum likelihood detection. For more information on the KBMOD algorithm see the following papers:
* [Fast Algorithms for Slow Moving Asteroids: Constraints on the Distribution of Kuiper Belt Objects](https://ui.adsabs.harvard.edu/abs/2019AJ....157..119W/abstract) by Whidden et. al. (2019)
* [Sifting Through the Static: Moving Object Detection in Difference Images](https://arxiv.org/abs/2109.03296) by Smotherman et. al. (2021)

Current documentation, including instructions for installation, can be found [here](https://epyc.astro.washington.edu/~kbmod/).


## Requirements

To build `kbmod` The packages required to build the code are:
* Cuda Toolkit >= 8.0
* CMake >= 3.23

Ensure that the NVIDIA's `nvcc` compiler is available on your system, for example:
```
nvcc --version
```
It is possible that the compiler is installed but not discoverable. In that case add its location to `PATH`. For example, if using `bash`  do `export PATH=/path/to/cuda:$PATH`. The default location for CUDA Toolkit installation is usually `/usr/local/cuda-XY.Z**` where `XY.Z` represent the CUDA Toolkit version that was installed.    
If using `bash` add the appropriate command to `~/.bashrc` in order to avoid having to set it repeatedly.

If CUDA Toolkit is not available on your system follow their [official installation instructions](https://developer.nvidia.com/cuda-toolkit). Optionally, if you use Anaconda virtual environments, the CUDA Toolkit is also available as `conda install cudatoolkit-dev`. Depending on the version of drivers on your GPU, you might need to use an older cudatoolkit-dev version.

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

This builds the package and all the dependencies required to test, run KBMOD on images and read the results. To use the additional analysis tools available in the `analysis` module it is necessary to install additional dependencies:
```
pip install .[analysis]
```
Note, however, that some of the dependencies in the `analysis` module require packages and supplementary data that are not installed nor provided by KBMoD. 

To verify that the installation was successful run the tests:
```
cd tests/
python -m unittest
```

### For Developers

If you want to contribute to the development of KBMOD, it is recommended that you install it in editable mode:
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

If you want to build the documentation you must have pandoc which seems not installable by pip.
See [Pandoc](https://pandoc.org/installing.html), or if you are using conda:
```
conda install pandoc
```
Building the documentation in docs/build/html using sphinx:
```
pip install .[docs]
sphinx-build -t html docs/source docs/build
```
Or you can use the make to call sphinx:
```
cd docs
make clean html
```
## Usage

A short example injecting a simulated object into a stack of images, and then recovering it. This example is also included in `tests/test_readme_example.py`.

```python

from kbmod.search import StackSearch, Trajectory
from kbmod.trajectory_generator import KBMODV1Search
from kbmod.fake_data.fake_data_creator import *

# Create fake data with ten 512x512 pixel images and starting at MJD of 57130.2.
fake_times = create_fake_times(10, t0=57130.2)
ds = FakeDataSet(512, 512, fake_times)

# Insert an artificial object with starting position x=2, y=0,
# velocity vx=10.7, vy=15.3, and flux = 275.0.
trj = Trajectory(x=2, y=0, vx=10.7, vy=15.3, flux=275.0)
ds.insert_object(trj)

# Generate a set of trajectories to test from each pixel.
gen = KBMODV1Search(
    5,  # Number of search velocities to try (0, 0.8, 1.6, 2.4, 3.2)
    0,  # The minimum search velocity to test (inclusive)
    4,  # The maximum search velocity to test (exclusive)
    5,  # Number of search angles to try (-0.1, -0.06, -0.02, 0.02, 0.6)
    -0.1,  # The minimum search angle to test (inclusive)
    0.1,  # The maximum search angle to test (exclusive)
)
candidates = [trj for trj in gen]

# Do the actual search (on CPU).  This requires passing in the science
# images, the variance images, the PSF information, and the times.
search = StackSearch(
    ds.stack_py.sci,
    ds.stack_py.var,
    ds.stack_py.psfs,
    ds.stack_py.zeroed_times,
)
search.set_min_obs(7)
search.search_all(candidates, False)

# Get the top 10 results.
results = search.get_results(0, 10)
print(results)
```

## Reference

* [API Reference](notebooks/Kbmod_Reference.ipynb)
* [Search Demo](notebooks/KBMOD_Demo.ipynb)

## License

The software is open source and available under the BSD license.

## Acknowledgements

This project is supported by Schmidt Sciences.

The team acknowledges support from the DIRAC Institute in the Department of Astronomy at the University of Washington. The DIRAC Institute is supported through generous gifts from the Charles and Lisa Simonyi Fund for Arts and Sciences, and the Washington Research Foundation.
