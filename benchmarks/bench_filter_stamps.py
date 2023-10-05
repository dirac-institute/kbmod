import timeit
import numpy as np

from kbmod.search import *


def setup_coadd_stamp(params):
    """Create a coadded stamp to test with a single bright spot
    slightly off center.

    Parameters
    ----------
    params : `StampParameters`
        The parameters for stamp generation and filtering.

    Returns
    -------
    stamp : `RawImage`
        The coadded stamp.
    """
    stamp_width = 2 * params.radius + 1

    stamp = RawImage(stamp_width, stamp_width)
    stamp.set_all(0.5)

    # Insert a flux of 50.0 and apply a PSF.
    flux = 50.0
    p = PSF(1.0)
    psf_dim = p.get_dim()
    psf_rad = p.get_radius()
    for i in range(psf_dim):
        for j in range(psf_dim):
            stamp.set_pixel(
                (params.radius - 1) - psf_rad + i,  # x is one pixel off center
                params.radius - psf_rad + j,  # y is centered
                flux * p.get_value(i, j),
            )

    return stamp


def run_benchmark(stamp_radius=10):
    params = StampParameters()
    params.radius = stamp_radius
    params.do_filtering = True
    params.stamp_type = StampType.STAMP_MEAN
    params.center_thresh = 0.03
    params.peak_offset_x = 1.5
    params.peak_offset_y = 1.5
    params.m01_limit = 0.6
    params.m10_limit = 0.6
    params.m11_limit = 2.0
    params.m02_limit = 35.5
    params.m20_limit = 35.5

    # Create the stamp.
    stamp = setup_coadd_stamp(params)

    # Create an empty search stack.
    im_stack = ImageStack([])
    search = StackSearch(im_stack)

    # Do three timing runs and use the mean of the time taken.
    tmr = timeit.Timer(stmt="search.filter_stamp(stamp, params)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


if __name__ == "__main__":
    for r in [5, 10, 20]:
        res_time = run_benchmark(r)
        print(f"Stamp Radius={r} -> Ave Time={res_time}")
