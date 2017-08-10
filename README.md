# KBMOD (Kernel-Based Moving Object Detection)

[![Build Status](https://travis-ci.org/DiracInstitute/kbmod.svg?branch=master)](https://travis-ci.org/DiracInstitute/kbmod) [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

A Maximum Likelihood detection algorithm for moving astronomical objects.

KBMOD is a set of Python tools to search astronomical images for moving
objects based upon method of maximum likelihood detection.

## Requirements

The packages required to run the code are:

* Numpy
* Scipy
* Scikit-learn
* Matplotlib
* Astropy
* Cuda 8.0
* CMake

## Setup

After cloning the repository and moving into it's directory, run ```git submodule init``` followed by ```git submodule update``` to also clone in pybind11

Build the python module by running ```cmake ./``` followed by ```make``` from inside the search/pybinds folder 

Then in the root directory run ```source setup.bash```
to append the library to the python path


## Example

See the example [ipython notebook](https://github.com/jbkalmbach/kbmod/blob/master/notebooks/kbmod_demo.ipynb).

## License

The software is open source and available under the BSD license.
