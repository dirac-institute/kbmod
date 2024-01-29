"""This file tests the functions in src/kbmod/fake_orbits. Since those files
include optional dependencies and data files, we do not include them as
part of the standard test suite.

TODO: Set up the test suite to pull in the optional data files.
"""

from astropy.wcs import WCS

import kbmod.search as kb

from kbmod.configuration import SearchConfiguration
from kbmod.fake_orbits.insert_fake_orbit import insert_fake_orbit_into_work_unit
from kbmod.fake_orbits.pyoorb_helper import PyoorbOrbit
from kbmod.work_unit import WorkUnit

# Set the image parameters.
width = 500
height = 700
num_images = 20
obs_cadence = 5

# If you have manually downloaded the data files, you need to add:
# PyoorbOrbit.specify_data_path(my_data_dir)
my_orb = PyoorbOrbit.from_kepler_elements(40.0, 0.05, 0.1, 0.0, 0.0, 0.0)

# Create time steps spaced 'obs_cadence' days apart.
times = [55144.5 + obs_cadence * i for i in range(num_images)]

# Predict the observations in (RA, dec) at those times.
# Using Gemini South as an observatory.
predicted_pos = my_orb.get_ephemerides(times, "I11")

# Create a fake WCS centered near where the object will be on 11th
# observation.
header_dict = {
    "WCSAXES": 2,
    "CTYPE1": "RA---TAN-SIP",
    "CTYPE2": "DEC--TAN-SIP",
    "CRVAL1": predicted_pos[10].ra.value,
    "CRVAL2": predicted_pos[10].dec.value,
    "CRPIX1": width / 2,
    "CRPIX2": height / 2,
    "CDELT1": 0.001,
    "CDELT2": 0.001,
}
wcs = WCS(header_dict)

# Create the fake images. Set slightly different PSFs per image.
images = [None] * num_images
for i in range(num_images):
    images[i] = kb.LayeredImage(
        ("layered_test_%i" % i),
        width,
        height,
        2.0,  # noise_level
        4.0,  # variance
        times[i],
        kb.PSF(1.0 + 0.01 * i),
    )
im_stack = kb.ImageStack(images)

# Create the WorkUnit with a default config.
config = SearchConfiguration()
work = WorkUnit(im_stack, config, wcs)

# Compute the pixel positions
results = insert_fake_orbit_into_work_unit(work, my_orb, 100.0, "I11")
print("RESULTS:")
for i in range(len(results)):
    # Do basic bounds checking.
    if results[i] is not None:
        assert results[i][0] >= 0
        assert results[i][1] >= 0
        assert results[i][0] < width
        assert results[i][1] < height
    print(f"    {i}: {results[i]}")
