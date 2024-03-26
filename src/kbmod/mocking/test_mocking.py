import timeit

from .fits import EmptyFits, SimpleFits, DECamImdiffs
from kbmod import ImageCollection
import matplotlib.pyplot as plt

start = timeit.default_timer()
fits_files = DECamImdiffs().mock(50)
print(f"DECam Imdiff Fits: {timeit.default_timer() - start} seconds")
del fits_files

start = timeit.default_timer()
fits_files = EmptyFits().mock(50)
print(f"EmptyFits: {timeit.default_timer() - start} seconds")
del fits_files

start = timeit.default_timer()
fits_files = SimpleFits.from_defaults(
    shape=(1000, 1000),
    add_static_sources=False,
    add_moving_objects=True,
    noise=100,
    noise_std=20
).mock(50)
print(f"SimpleFits: {timeit.default_timer() - start} seconds")


start = timeit.default_timer()
ic = ImageCollection.fromTargets(fits_files, force="TestDataStd")
print(f"Image collection took: {timeit.default_timer() - start} seconds")
print(ic)

_ = [plt.imsave(f"/home/dino/Downloads/tmp_fits/{i}.png", h[1].data, vmin=50, vmax=500) for i, h in enumerate(fits_files)]


breakpoint()
