import timeit
import numpy as np

from kbmod.filters.stamp_filters import *
from kbmod.result_list import ResultRow
from kbmod.search import ImageStack, PSF, RawImage, StackSearch, StampCreator, StampParameters, StampType, Trajectory


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


def run_search_benchmark(params):
    stamp = setup_coadd_stamp(params)

    # Create an empty search stack.
    im_stack = ImageStack([])
    search = StackSearch(im_stack)
    sc = StampCreator()

    # Do the timing runs.
    tmr = timeit.Timer(stmt="sc.filter_stamp(stamp, params)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def run_row_benchmark(params, create_filter=""):
    stamp = setup_coadd_stamp(params)
    row = ResultRow(Trajectory(), 10)
    row.stamp = np.array(stamp.get_all_pixels())

    filt = eval(create_filter)

    # Do the timing runs.
    full_cmd = "filt.keep_row(row)"
    tmr = timeit.Timer(stmt="filt.keep_row(row)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def run_all_benchmarks():
    params = StampParameters()
    params.radius = 5
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

    print(" Rad |       Method       |    Time")
    print("-" * 40)
    for r in [2, 5, 10, 20]:
        params.radius = r

        res_time = run_search_benchmark(params)
        print(f"  {r:2d} | C++ (all)          | {res_time:10.7f}")

        res_time = run_row_benchmark(params, f"StampPeakFilter({r}, 1.5, 1.5)")
        print(f"  {r:2d} | StampPeakFilter    | {res_time:10.7f}")

        res_time = run_row_benchmark(params, f"StampMomentsFilter({r}, 0.6, 0.6, 2.0, 35.5, 35.5)")
        print(f"  {r:2d} | StampMomentsFilter | {res_time:10.7f}")

        res_time = run_row_benchmark(params, f"StampCenterFilter({r}, False, 0.03)")
        print(f"  {r:2d} | StampCenterFilter  | {res_time:10.7f}")
        print("-" * 40)


if __name__ == "__main__":
    run_all_benchmarks()
