import numpy as np
from astropy.time import Time
from astropy.table import QTable, vstack


__all__ = [
    "gen_random_static_source_catalog",
    "gen_random_moving_object_catalog"
]


def gen_random_static_source_catalog(n, param_ranges, seed=None):
    cat = QTable()
    rng = np.random.default_rng(seed)

    for param_name, (lower, upper) in param_ranges.items():
        cat[param_name] = rng.uniform(lower, upper, n)

    if "stddev" in param_ranges:
        cat["x_stddev"] = cat["stddev"]
        cat["y_stddev"] = cat["stddev"]

    # conversion assumes a gaussian
    if "flux" in param_ranges and "amplitude" not in param_ranges:
        xstd = cat["x_stddev"] if "x_stddev" in cat.colnames else 1
        ystd = cat["y_stddev"] if "y_stddev" in cat.colnames else 1

        cat["amplitude"] = cat["flux"] / (2.0 * np.pi * xstd * ystd)

    return cat


def gen_random_moving_object_catalog(n, param_ranges, dt, seed=None):
    obj_cat = gen_random_static_source_catalog(n, param_ranges, seed)
    return obj_cat
#    cat = None
#    for i, t in enumerate(obstimes):
#        if i == 0:
#            cat = obj_cat.copy()
#            continue
#
#        obj_cat["x_mean"] += obj_cat["vx"]*dt
#        vstack(cat, obj_cat)
#
#    return cat


