import abc

import numpy as np
from astropy.time import Time
from astropy.table import QTable, vstack


__all__ = [
    "gen_catalog",
    "CatalogFactory",
    "SimpleSourceCatalog",
    "SimpleObjectCatalog",
]


def gen_catalog(n, param_ranges, seed=None):
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


def gen_source_catalog(n, param_ranges, seed=None):


class CatalogFactory(abc.ABC):
    @abc.abstractmethod
    def gen_realization(self, *args, t=None, dt=None, **kwargs):
        raise NotImplementedError()

    def mock(self, *args, **kwargs):
        return self.gen_realization(self, *args, **kwargs)


class SimpleSourceCatalog(CatalogFactory):
    base_param_ranges = {
        "amplitude": [500, 2000],
        "x_mean": [0, 4096],
        "y_mean": [0, 2048],
        "x_stddev": [1, 7],
        "y_stddev": [1, 7],
        "theta": [0, np.pi],
    }

    def __init__(self, table, return_copy=False):
        self.table = table
        self.return_copy = return_copy

    @classmethod
    def from_params(cls, n=100, param_ranges=None):
        param_ranges = {} if param_ranges is None else param_ranges
        tmp = cls.base_param_ranges.copy()
        tmp.update(param_ranges)
        return cls(gen_catalog(n, tmp))

    def gen_realization(self, *args, t=None, dt=None, **kwargs):
        if self.return_copy:
            return self.table.copy()
        return self.table

    def mock(self, n=1):
        if n == 1:
            return self.table
        return [self.table for i in range(n)]


class SimpleObjectCatalog(CatalogFactory):
    base_param_ranges = {
        "amplitude": [50, 500],
        "x_mean": [0, 4096],
        "y_mean": [0, 2048],
        "vx": [500, 5000],
        "vy": [500, 5000],
        "stddev": [1, 2],
        "theta": [0, np.pi],
    }

    dtype = np.dtype([
        ("amplitude", np.float32),
        ("x_mean", np.float32),
        ("y_mean", np.float32),
        ("vx", np.float32),
        ("vy", np.float32),
        ("x_stddev", np.float32),
        ("y_stddev", np.float32),
        ("thetae", np.float32)
    ]

    def __init__(self, table, obstime=None):
        self.table = table
        self._realization = table.copy()
        self.obstime = 0 if obstime is None else obstime

    @classmethod
    def from_params(cls, n=10, param_ranges=None):
        param_ranges = {} if param_ranges is None else param_ranges
        tmp = cls.base_param_ranges.copy()
        tmp.update(param_ranges)
        return cls(gen_catalog(n, tmp))

    def gen_realization(self, t=None, dt=None, **kwargs):
        if t is None and dt is None:
            return self._realization

        dt = dt if t is None else t - self.obstime
        self._realization["x_mean"] += self._realization["vx"] * dt
        self._realization["y_mean"] += self._realization["vy"] * dt
        return self._realization

    def mock(self, n=1, dt=0.001):
        if n == 1:
            return self.gen_realization()

