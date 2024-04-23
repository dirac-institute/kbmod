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

    if "stddev" in param_ranges:
        cat["x_stddev"] = cat["stddev"]
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
    return_copy = False
    seed = None
    n = 100
    param_ranges = {}


class SimpleCatalog(CatalogFactory):
    default_config = SimpleCatalogConfig

    def __init_from_table(self, table, config=None, **kwargs):
        config = self.default_config(config=config, **kwargs)
        config.n = len(table)
        params = {}
        for col in table.keys():
            params[col] = (table[col].min(), table[col].max())
        config.param_ranges.update(params)
        return config, table

    def __init_from_config(self, config, **kwargs):
        config = self.default_config(config=config, method="subset", **kwargs)
        table = gen_catalog(config.n, config.param_ranges, config.seed)
        return config, table

    def __init_from_ranges(self, **kwargs):
        param_ranges = kwargs.pop("param_ranges", None)
        if param_ranges is None:
            param_ranges = {k: v for k, v in kwargs.items() if k in self.default_config.param_ranges}
            kwargs = {k: v for k, v in kwargs.items() if k not in self.default_config.param_ranges}

        config = self.default_config(**kwargs, method="subset")
        config.param_ranges.update(param_ranges)
        return self.__init_from_config(config=config)

    def __init__(self, table=None, config=None, **kwargs):
        if table is not None:
            config, table = self.__init_from_table(table, config=config, **kwargs)
        elif isinstance(config, Config):
            config, table = self.__init_from_config(config=config, **kwargs)
        elif isinstance(config, dict) or kwargs:
            config = {} if config is None else config
            config, table = self.__init_from_ranges(**{**config, **kwargs})
        else:
            raise ValueError(
                "Expected table or config, or keyword arguments of expected "
                f"catalog value ranges, got:\n table={table}\n config={config} "
                f"\n kwargs={kwargs}"
            )

        self.config = config
        self.table = table
        self.current = 0

    @classmethod
    def from_config(cls, config, **kwargs):
        config = cls.default_config(config=config, method="subset", **kwargs)
        return cls(gen_catalog(config.n, config.param_ranges, config.seed), config=config)

    @classmethod
    def from_ranges(cls, n=None, config=None, **kwargs):
        config = cls.default_config(n=n, config=config, method="subset")
        config.param_ranges.update(**kwargs)
        return cls.from_config(config)

    @classmethod
    def from_table(cls, table):
        config = cls.default_config()
        config.n = len(table)
        params = {}
        for col in table.keys():
            params[col] = (table[col].min(), table[col].max())
        config["param_ranges"] = params
        return cls(table, config=config)

    def mock(self):
        self.current += 1
        if self.config.return_copy:
            return self.table.copy()
        return self.table


class SourceCatalogConfig(SimpleCatalogConfig):
    param_ranges = {
        "amplitude": [1., 10.],
        "x_mean": [0., 4096.],
        "y_mean": [0., 2048.],
        "x_stddev": [1., 3.],
        "y_stddev": [1., 3.],
        "theta": [0., np.pi],
    }


class SourceCatalog(SimpleCatalog):
    default_config = SourceCatalogConfig


class ObjectCatalogConfig(SimpleCatalogConfig):
    param_ranges = {
        "amplitude": [0.1, 3.0],
        "x_mean": [0., 4096.],
        "y_mean": [0., 2048.],
        "vx": [500., 1000.],
        "vy": [500., 1000.],
        "stddev": [0.25, 1.5],
        "theta": [0., np.pi],
    }


class ObjectCatalog(SimpleCatalog):
    default_config = ObjectCatalogConfig

    def __init__(self, table=None, obstime=None, config=None, **kwargs):
        # put return_copy into kwargs to override whatever user might have
        # supplied, and to guarantee the default is overriden
        kwargs["return_copy"] = True
        super().__init__(table=table, config=config, **kwargs)
        self._realization = self.table.copy()
        self.obstime = 0 if obstime is None else obstime

    def reset(self):
        self.current = 0
        self._realization = self.table.copy()

    def gen_realization(self, t=None, dt=None, **kwargs):
        if t is None and dt is None:
            return self._realization

        dt = dt if t is None else t - self.obstime
        self._realization["x_mean"] += self.table["vx"] * dt
        self._realization["y_mean"] += self.table["vy"] * dt
        return self._realization

    def mock(self, n=1, **kwargs):
        breakpoint()
        if n == 1:
            data = self.gen_realization(**kwargs)
            self.current += 1
        else:
            data = []
            for i in range(n):
                data.append(self.gen_realization(**kwargs).copy())
                self.current += 1

        return data
