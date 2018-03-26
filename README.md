<img src="https://gist.githubusercontent.com/PWhiddy/d42e66a9dd8e4af205a706f388a90ed4/raw/ae5bb87ada12538289852b58ba8e54b564a81584/kbmod.svg?sanitize=true" alt="logo" width="400" height="160"/>

An image processing framework for moving object detection implemented with GPUs.  
Based on a Maximum Likelihood detection algorithm for moving astronomical objects.

[![Build Status](https://travis-ci.org/DiracInstitute/kbmod.svg?branch=master)](https://travis-ci.org/DiracInstitute/kbmod) [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)



KBMOD is a set of Python tools to search astronomical images for moving
objects based upon method of maximum likelihood detection.

## Setup

**Requirements**

The packages required to build the code are:

* python3 development headers
* Scipy (Numpy, Matplotlib)
* Scikit-learn
* Cuda 8.0
* CMake 3

**To install:**  
Open search/pybinds/CmakeLists.txt and verify in the "include_directories" section that the paths to the python headers and to the cuda installation are correct. 

Then run 
```source install.sh```
This will build the python library and run the tests.

If you log out, next time run
```source setup.bash```
to reappend the library to the python path

## Usage

A short example injecting a simulated object into a stack of images, and then recovering it.

```python

from kbmodpy import kbmod as kb
import numpy as np

# Create a point spread function
psf = kb.psf(1.5)

# load images from list of file paths
imgs =  [ kb.layered_image(file) for file in example_files ]

# Specify an artificial object
flux = 175.0
position = (100.7, 150.3)
velocity = (50, 35)

# Inject object into images
for im in imgs:
    im.add_object(position[0]+im.get_time()*velocity[0], 
                  position[1]+im.get_time()*velocity[1], 
                  flux, psf)

# Recover the object by searching a wide region
velocity_guess = (40, 40)
radius = 20
min_lh = 9.0
min_obs = 10
stack = kb.image_stack(imgs)
search = kb.stack_search(stack, psf)
results = search.region_search(*velocity_guess, radius, min_lh, min_obs)

```

[Short Demonstration](notebooks/Quick_Test.ipynb)

[Processing Real Images](notebooks/HITS_Main_Belt_Comparison.ipynb)

## Reference

[API Reference](notebooks/Kbmod_Reference.ipynb).

## License

The software is open source and available under the BSD license.
