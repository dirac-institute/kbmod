import timeit

from .fits import EmptyFits, SimpleFits, DECamImdiffs
from kbmod import ImageCollection
import matplotlib.pyplot as plt

start = timeit.default_timer()
#fits_files = DECamImdiffs().mock(50)
print(f"DECam Imdiff Fits: {timeit.default_timer() - start} seconds")
#del fits_files

start = timeit.default_timer()
#fits_files = EmptyFits().mock(50)
print(f"EmptyFits: {timeit.default_timer() - start} seconds")
#del fits_files

start = timeit.default_timer()
fits_files = SimpleFits.from_defaults(
    shape=(500, 500),
    add_static_sources=False,
    add_moving_objects=True,
    noise=1000,
).mock(50)
print(f"SimpleFits: {timeit.default_timer() - start} seconds")

#fits_factory = SimpleFits.from_defaults(add_moving_objects=False)
#print(f"Fits factory took: {timeit.default_timer() - start} seconds")

#fits_files = [fits_factory.mock() for i in range(50)]
#print(f"Generated 100 images in: {timeit.default_timer() - start} seconds")

start = timeit.default_timer()
ic = ImageCollection.fromTargets(fits_files, force="TestDataStd")
print(f"Image collection took: {timeit.default_timer() - start} seconds")
print(ic)

_ = [plt.imsave(f"/home/dino/Downloads/tmp_fits/{i}.png", h[1].data) for i, h in enumerate(fits_files)]


breakpoint()
