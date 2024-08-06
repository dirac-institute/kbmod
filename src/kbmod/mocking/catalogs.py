import abc

import numpy as np
from astropy.table import QTable
from .config import Config


__all__ = [
    "gen_catalog",
    "CatalogFactory",
    "SimpleCatalog",
    "SourceCatalogConfig",
    "SourceCatalog",
    "ObjectCatalogConfig",
    "ObjectCatalog",
]


def gen_catalog(n, param_ranges, seed=None):
    cat = QTable()
    rng = np.random.default_rng(seed)

    for param_name, (lower, upper) in param_ranges.items():
        cat[param_name] = rng.uniform(lower, upper, n)

    if "x_stddev" not in param_ranges:
        cat["x_stddev"] = cat["stddev"]
    if "y_stddev" not in param_ranges:
        cat["y_stddev"] = cat["stddev"]

    # conversion assumes a gaussian
    if "flux" in param_ranges and "amplitude" not in param_ranges:
        xstd = cat["x_stddev"] if "x_stddev" in cat.colnames else 1.0
        ystd = cat["y_stddev"] if "y_stddev" in cat.colnames else 1.0

        cat["amplitude"] = cat["flux"] / (2.0 * np.pi * xstd * ystd)

    return cat


class CatalogFactory(abc.ABC):
    @abc.abstractmethod
    def mock(self, *args, **kwargs):
        raise NotImplementedError()


class SimpleCatalogConfig(Config):
    mode = "static"
    return_copy = False
    seed = None
    n = 100
    param_ranges = {}


class SimpleCatalog(CatalogFactory):
    default_config = SimpleCatalogConfig

    def __init__(self, config, table, **kwargs):
        config = self.default_config(**kwargs)
        self.config = config
        self.table = table
        self.current = 0

    @classmethod
    def from_config(cls, config, **kwargs):
        config = cls.default_config(config=config, **kwargs)
        table = gen_catalog(config.n, config.param_ranges, config.seed)
        return cls(config, table)

    @classmethod
    def from_defaults(cls, param_ranges=None, **kwargs):
        config = cls.default_config(**kwargs)
        if param_ranges is not None:
            config.param_ranges.update(param_ranges)
        return cls.from_config(config)

    @classmethod
    def from_table(cls, table, **kwargs):
        config = cls.default_config(**kwargs)
        config.n = len(table)
        params = {}
        for col in table.keys():
            params[col] = (table[col].min(), table[col].max())
        config["param_ranges"] = params
        return cls(config, table)

    def mock(self):
        self.current += 1
        if self.config.return_copy:
            return self.table.copy()
        return self.table


class SourceCatalogConfig(SimpleCatalogConfig):
    param_ranges = {
        "amplitude": [1.0, 10.0],
        "x_mean": [0.0, 4096.0],
        "y_mean": [0.0, 2048.0],
        "x_stddev": [1.0, 3.0],
        "y_stddev": [1.0, 3.0],
        "theta": [0.0, np.pi],
    }


class SourceCatalog(SimpleCatalog):
    default_config = SourceCatalogConfig


class ObjectCatalogConfig(SimpleCatalogConfig):
    mode = "progressive"  # folding
    param_ranges = {
        "amplitude": [0.1, 3.0],
        "x_mean": [0.0, 4096.0],
        "y_mean": [0.0, 2048.0],
        "vx": [500.0, 1000.0],
        "vy": [500.0, 1000.0],
        "stddev": [0.25, 1.5],
        "theta": [0.0, np.pi],
    }


class ObjectCatalog(SimpleCatalog):
    default_config = ObjectCatalogConfig

    def __init__(self, config, table, **kwargs):
        # Obj cat always has to return a copy
        kwargs["return_copy"] = True
        super().__init__(config, table, **kwargs)
        self._realization = self.table.copy()
        self.mode = self.config.mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if val == "folding":
            self._gen_realization = self.fold
        elif val == "progressive":
            self._gen_realization = self.next
        elif val == "static":
            self._gen_realization = self.static
        else:
            raise ValueError(
                "Unrecognized object catalog mode. Expected 'static', "
                f"'progressive', or 'folding', got {val} instead."
            )
        self._mode = val

    def reset(self):
        self.current = 0
        self._realization = self.table.copy()

    def static(self, **kwargs):
        return self.table.copy()

    def next(self, dt, **kwargs):
        self._realization["x_mean"] = self.table["x_mean"] + self.current*self._realization["vx"] * dt
        self._realization["y_mean"] = self.table["y_mean"] + self.current*self._realization["vy"] * dt
        self.current += 1
        return self._realization.copy()

    def fold(self, t, **kwargs):
        self._realization = self.table[self.table["obstime"] == t]
        self.current += 1
        return self._realization.copy()

    def mock(self, n=1, **kwargs):
        data = []

        if self.mode == "folding":
            for t in kwargs["t"]:
                data.append(self.fold(t=t))
        else:
            for i in range(n):
                data.append(self._gen_realization(**kwargs))

        return data
