<img src="https://gist.githubusercontent.com/PWhiddy/d42e66a9dd8e4af205a706f388a90ed4/raw/ae5bb87ada12538289852b58ba8e54b564a81584/kbmod.svg?sanitize=true" alt="logo" width="500" height="200"/>

A kernel-Based Moving Object Detection image processing framework with CUDA  
Based on a Maximum Likelihood detection algorithm for moving astronomical objects.

[![Build Status](https://travis-ci.org/DiracInstitute/kbmod.svg?branch=master)](https://travis-ci.org/DiracInstitute/kbmod) [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)



KBMOD is a set of Python tools to search astronomical images for moving
objects based upon method of maximum likelihood detection.

## Setup

**Requirements**

The packages required to build the code are:

* python3-dev
* Scipy (Numpy, Matplotlib)
* Scikit-learn
* Astropy
* Cuda 8.0
* CMake

**To install:**  
```source install.sh```
This will build the python library and run the tests.

If you log out, next time run
```source setup.bash```
to reappend the library to the python path

## Useage

[Short Demonstration](notebooks/Quick_Test.ipynb)

[Processing Real Images](notebooks/HITS_Main_Belt_Comparison.ipynb)

## Reference

[API Reference](notebooks/Kbmod_Reference.ipynb).

## License

The software is open source and available under the BSD license.
