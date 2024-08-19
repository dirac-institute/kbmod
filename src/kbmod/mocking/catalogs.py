import abc

import numpy as np
from astropy.table import QTable
from astropy.coordinates import SkyCoord

from .config import Config


__all__ = [
    "gen_random_catalog",
    "SimpleCatalog",
    "SourceCatalogConfig",
    "SourceCatalog",
    "ObjectCatalogConfig",
    "ObjectCatalog",
]



def expand_gaussian_cols(cat):
    """Expands columns ``flux`` and ``stddev`` into ``amplitude``, ``x_stddev``
    and ``y_stddev`` assuming the intended catalog model is a symmetric 2D
    Gaussian.

    Amplitude is caluclated as:
        A  = flux/(2*pi*sigma_x*sigma_y)

    Parameters
    ----------
    cat : `astropy.table.Table`
        A catalog of simplified model parameters.

    Returns
    ------
    expanded : `astropy.table.Table`
        A catalog of AstroPy model parameters.
    """
    if "x_stddev" not in cat.columns and "stddev" in cat.columns:
        cat["x_stddev"] = cat["stddev"]
    if "y_stddev" not in cat.columns and "stddev" in cat.columns:
        cat["y_stddev"] = cat["stddev"]

    if "flux" in cat.columns and "amplitude" not in cat.columns:
        cat["amplitude"] = cat["flux"] / (2.0 * np.pi * xstd * ystd)

    return cat


def gen_random_catalog(n, param_ranges, seed=None, assume_gaussian=True):
    """Generates a random catalog of parameters of n sources based on
    AstroPy's 2D models.

    The object parameters are specified as a dict where keys become columns and
    the values represent the range from which each parameter is uniformly
    randomly drawn from.

    If parameter ranges contain ``flux``, a column ``amplitude`` will be added
    which value will be calculated assuming a 2D Gaussian model. If ``stddev``
    column is preset, then ``x_stddev`` and ``y_stddev`` columns are added,
    assuming the model intended to be used with the catalog is a symmetrical
    2D Gaussian.

    Parameters
    ----------
    n : `int`
        Number of objects to create
    param_ranges : `dict`
        Dictionary whose keys become columns of the catalog and which values
        define the ranges from which the catalog values are drawn from.
    seed : `int`
        NumPy's random number generator seed.
    assume_gaussian : `bool`
        Assume the catalog is intended for use with a 2D Gaussian model and
        expand ``flux`` and ``stddev`` columns appropriately. See
        `expand_gaussian_cols`/`

    Returns
    -------
    catalog : `astropy.table.Table`
        Catalog

    Examples
    --------
    >>> import kbmod.mocking as kbmock
    >>> kbmock.gen_random_catalog(3, {"amplitude": [0, 3], "x": [3, 6], "y": [6, 9]}, seed=100)
    <QTable length=3>
        amplitude              x                 y
         float64            float64           float64
    ------------------ ----------------- -----------------
    2.504944891506027 3.128854712082634 8.370789493256163
    1.7896620809036619 5.920963185318643  8.73101814388636
    0.8665897250736108 4.789415112194066 8.064463342775236
    """
    cat = QTable()
    rng = np.random.default_rng(seed)

    for param_name, (lower, upper) in param_ranges.items():
        cat[param_name] = rng.uniform(lower, upper, n)

    if assume_gaussian:
        cat = expand_gaussian_cols(cat)

    return cat


class SimpleCatalogConfig(Config):
    """A simple catalog configuration."""

    mode = "static"
    """Static, progressive or folding; static catalogs remain the same for every
    realization, progressive catalogs modify values of realized catalog columns
    and folding catalogs select rows from a larger catalog to return as a realization.
    """

    kind = "pixel"
    """Either ``pixel`` or ``world``. The kind of position coordinates encoded
    by the catalog. On-sky, world, coordinates require a list of WCSs to be given
    to the mocking method"""

    return_copy = False
    """For static catalogs, return a reference to the underlying catalog or
    a copy that can be modified."""

    seed = None
    """Random number generator seed. When `None` a different seed is used for
    every catalog factory."""

    n = 100
    """Number of objects to generate."""

    param_ranges = {}
    """Default parameter ranges of the catalog values and columns that will be
    generated."""

    pix_pos_cols = ["x_mean", "y_mean"]
    """Which """

    pix_vel_cols = ["vx", "vy"]

    world_pos_cols = ["ra_mean", "dec_mean"]

    world_vel_cols = ["v_ra", "v_dec"]

    fold_col = "obstime"


class SimpleCatalog:
    """Default base class for mocked catalogs factory.

    This base class will always generate empty catalogs and is intended to be
    inherited from.

    A catalog is an `astropy.table.Table` of model values (position, amplitude
    etc.) of sources. A catalog factory mock realizations of catalogs. There
    are 2 ``kind``s of catalogs that can be mocked in 3 different ``modes``.

    In progressive catalogs mode an existing base catalog template column
    values are incremented, or otherwise updated for each new realization.
    Static catalogs always return the same realization of a catalog. And the
    folding catalogs depend on an underlying larger catalog, from which they
    select which rows to return as a new realization. This is namely most
    appropriate for catalogs with timestamps, where a different realization of
    a catalog is returned per timestamp.
    When the catalog mode is ``folding`` the mocking method expects the values
    on which to fold on. By default, these are timestamps `t`.

    If the catalog coordinate kind is ``pixel``, then the positions are
    interpreted as pixel coordiantes. If the kind of catalog coordinates are
    ``world`` then the positions are interpreted as on-sky coordinates in
    decimal degrees and a list of WCSs is expected to be provided to the mocking
    method.

    Parameters
    ----------
    config : `SimpleCatalogConfig`
        Factory configuration.
    table : `astropy.table.Table`
        The catalog template from which new realizations will be generated.

    Attributes
    ----------
    config : `SimpleCatalogConfig`
       The instance-bound configuration of the factory
    table : `astropy.table.Table`
       The template catalog used to create new realizations.
    _realization : `astropy.table.Table`
       The last realization of a catalog.
    current : `int`
       The current iterator counter.

    Examples
    --------
    Directly instantiate a simple static catalog:

    >>> import kbmod.mocking as kbmock
    >>> table = kbmock.gen_random_catalog(3, {"A": [0, 1], "x": [1, 2], "y": [2, 3]}, seed=100)
    >>> f = kbmock.SimpleCatalog(table)
    >>> f.mock()
    [<QTable length=3>
            A                  x                  y
         float64            float64            float64
    ------------------ ------------------ ------------------
    0.8349816305020089 1.0429515706942114 2.7902631644187212
    0.5965540269678873 1.9736543951062142 2.9103393812954526
    0.2888632416912036 1.5964717040646885  2.688154447591745]

    Instantiating from factory methods will derive additional information
    regarding the catalog contents:

    >>> f2 = kbmock.SimpleCatalog.from_table(table)
    >>> f2.mock()
    [<QTable length=3>
            A                  x                  y
         float64            float64            float64
    ------------------ ------------------ ------------------
    0.8349816305020089 1.0429515706942114 2.7902631644187212
    0.5965540269678873 1.9736543951062142 2.9103393812954526
    0.2888632416912036 1.5964717040646885  2.688154447591745]

    >>> f2.config["param_ranges"]
    {'A': (0.2888632416912036, 0.8349816305020089), 'x': (1.0429515706942114, 1.9736543951062142), 'y': (2.688154447591745, 2.9103393812954526)}
    >>> f.config["param_ranges"]
    {}

    Folding catalogs just return subsets of the template catalog:

    >>> table["obstime"] = [1, 1, 2]
    >>> f = kbmock.SimpleCatalog(table, mode="folding")
    >>> f.mock(t=[1, 2])
    [<QTable length=2>
            A                  x                  y          obstime
         float64            float64            float64        int64
    ------------------ ------------------ ------------------ -------
    0.8349816305020089 1.0429515706942114 2.7902631644187212       1
    0.5965540269678873 1.9736543951062142 2.9103393812954526       1, <QTable length=1>
            A                  x                  y         obstime
         float64            float64            float64       int64
    ------------------ ------------------ ----------------- -------
    0.2888632416912036 1.5964717040646885 2.688154447591745       2]

    And progressive catalogs increment selected column values (note the velocities
    were assigned the default expected column names but positions weren't):

    >>> table["vx"] = [1, 1, 1]
    >>> table["vy"] = [10, 10, 10]
    >>> f = kbmock.SimpleCatalog(table, mode="progressive", pix_pos_cols=["x", "y"])
    >>> _ = f.mock(dt=1); f.mock(dt=1)
    [<QTable length=3>
            A                  x                  y             vx      vy
         float64            float64            float64       float64 float64
    ------------------ ------------------ ------------------ ------- -------
    0.8349816305020089 2.0429515706942114 12.790263164418722     1.0    10.0
    0.5965540269678873  2.973654395106214 12.910339381295453     1.0    10.0
    0.2888632416912036 2.5964717040646885 12.688154447591746     1.0    10.0]
    """
    default_config = SimpleCatalogConfig

    def __init__(self, table, config=None, **kwargs):
        self.config = self.default_config(config=config, **kwargs)
        self.table = table
        self.current = 0
        self._realization = self.table.copy()
        self.mode = self.config["mode"]
        self.kind = self.config["kind"]

    @classmethod
    def from_defaults(cls, param_ranges=None, **kwargs):
        """Create a catalog factory using its default config.

        Parameters
        ----------
        param_ranges : `dict`
            Default parameter ranges of the catalog values and columns that will be
            generated. See `gen_random_catalog`.
        kwargs : `dict`
            Any additional keyword arguments will be used to supplement or override
            any matching default configuration parameters.

        Returns
        -------
        factory : `SimpleCatalog`
            Simple catalog factory.
        """
        config = cls.default_config(**kwargs)
        if param_ranges is not None:
            config["param_ranges"].update(param_ranges)
        table = gen_random_catalog(config["n"], config["param_ranges"], config["seed"])
        return cls(table=table, config=config)

    @classmethod
    def from_table(cls, table, **kwargs):
        """Create a factory from a table template, deriving parameters, their
        value ranges and number of objects from the table.

        Optionally expands the given table columns assuming the intended source
        model is a 2D Gaussian.

        Parameters
        ----------
        table : `astropy.table.Table`
            Catalog template.
        kwargs : `dict`
            Any additional keyword arguments will be used to supplement or override
            any matching default configuration parameters.

        Returns
        -------
        factory : `SimpleCatalog`
            Simple catalog factory.
        """
        table = expand_gaussian_cols(table)

        config = cls.default_config(**kwargs)
        config["n"] = len(table)
        params = {}
        for col in table.keys():
            params[col] = (table[col].min(), table[col].max())
        config["param_ranges"] = params
        return cls(table=table, config=config)

    @property
    def mode(self):
        """Catalog mode, ``static``, ``folding`` or ``progressive``."""
        return self._mode

    @mode.setter
    def mode(self, val):
        if val == "folding":
            self._gen_realization = self.fold
            self.config["return_copy"] = True
        elif val == "progressive":
            self._gen_realization = self.next
            self.config["return_copy"] = True
        elif val == "static":
            self._gen_realization = self.static
        else:
            raise ValueError(
                "Unrecognized object catalog mode. Expected 'static', "
                f"'progressive', or 'folding', got {val} instead."
            )
        self._mode = val

    @property
    def kind(self):
        """Catalog coordinate kind, ``pixel`` or ``world``"""
        return self._kind

    @mode.setter
    def kind(self, val):
        if val == "pixel":
            self._cat_keys = self.config["pix_pos_cols"] + self.config["pix_vel_cols"]
        elif val == "world":
            self._cat_keys = self.config["world_pos_cols"] + self.config["world_vel_cols"]
        else:
            raise ValueError(
                "Unrecognized coordinate kind. Expected 'world' or 'pixel, got"
                f"{val} instead."
            )
        self._kind = val

    def reset(self):
        """Reset the iteration counter reset the realization to the initial one."""
        self.current = 0
        self._realization = self.table.copy()

    def static(self, **kwargs):
        """Return the initial template as a catalog realization.

        Returns
        -------
        catalog : `astropy.table.Table`
            Catalog realization.
        """
        self.current += 1
        if self.config["return_copy"]:
            return self.table.copy()
        return self.table

    def next(self, dt):
        """Return the next catalog realization by incrementing the position
        columns by the value of the velocity column and number of current `dt` steps.

        Parameters
        ----------
        dt : `float`
            Time increment of each step.

        Returns
        -------
        catalog : `astropy.table.Table`
            Catalog realization.
        """
        a, b, va, vb = self._cat_keys
        self._realization[a] = self.table[a] + self.current * self.table[va] * dt
        self._realization[b] = self.table[b] + self.current * self.table[vb] * dt
        self.current += 1
        return self._realization.copy()

    def fold(self, t, **kwargs):
        """Return the next catalog realization by selecting those rows that
        match the given parameter ``t``. By default the folding column is
        ``obstime``.

        Parameters
        ----------
        t : `float`
            Value which to select from template catalog.

        Returns
        -------
        catalog : `astropy.table.Table`
            Catalog realization.
        """
        self._realization = self.table[self.table[self.config["fold_col"]] == t]
        self.current += 1
        return self._realization.copy()

    def mock(self, n=1, dt=None, t=None, wcs=None):
        """Return the next realization(s) of the catalogs.

        Selects the appropriate mocking function. Ignores keywords not
        appropriate for use given some catalog generation method and coordinate
        kind.

        Parameters
        ----------
        n : `int`, optional
           Number of catalogs to mock. Default 1.
        dt : `float`, optional.
           Timestep between each step (arbitrary units)
        t : `list[float]` or `list[astropy.time.Time]`, optional
           Values on which to fold the template catalog.
        wcs : `list[astropy.wcs.WCS]`, optional
           WCS to use in conversion of on-sky coordinates to pixel coordinates,
           for each realization.
        """
        data = []

        if self.mode == "folding":
            for i, ts in enumerate(t):
                data.append(self.fold(t=ts))
        else:
            for i in range(n):
                data.append(self._gen_realization(dt=dt))

        if self.kind == "world":
            racol, deccol = self.config["world_pos_cols"]
            xpixcol, ypixcol = self.config["pix_pos_cols"]
            for cat, w in zip(data, wcs):
                x, y = w.world_to_pixel(SkyCoord(ra=cat[racol], dec=cat[deccol], unit="deg"))
                cat[xpixcol] = x
                cat[ypixcol] = y

        return data


class SourceCatalogConfig(SimpleCatalogConfig):
    """Source catalog config.

    Assumes sources are static, asymmetric 2D Gaussians.

    Parameter ranges
    ----------------
    amplitude : [1, 10]
        Amplitude of the model.
    x_mean : [0, 4096]
        Real valued x coordinate of the object's centroid.
    y_mean : [0, 2048]
        Real valued y coordinate of the object's centroid.
    x_stddev : [1, 3]
        Real valued standard deviation of the model distribution, in x.
    y_stddev : [1, 3]
        Real valued standard deviation of the model distribution, in y.
    theta : `[0, np.pi]`
        Rotation of the model's covariance matrix, increases counterclockwise.
        In radians.
    """
    param_ranges = {
        "amplitude": [1.0, 10.0],
        "x_mean": [0.0, 4096.0],
        "y_mean": [0.0, 2048.0],
        "x_stddev": [1.0, 3.0],
        "y_stddev": [1.0, 3.0],
        "theta": [0.0, np.pi],
    }


class SourceCatalog(SimpleCatalog):
    """A static catalog representing stars and galaxies.

    Coordinates defined in pixel space.
    """
    default_config = SourceCatalogConfig


class ObjectCatalogConfig(SimpleCatalogConfig):
    """Object catalog config.

    Assumes objects are symmetric 2D Gaussians moving in a linear fashion.

    Parameter ranges
    ----------------
    amplitude : [1, 10]
        Amplitude of the model.
    x_mean : [0, 4096]
        Real valued x coordinate of the object's centroid.
    y_mean : [0, 2048]
        Real valued y coordinate of the object's centroid.
    x_stddev : [1, 3]
        Real valued standard deviation of the model distribution, in x.
    y_stddev : [1, 3]
        Real valued standard deviation of the model distribution, in y.
    theta : `[0, np.pi]`
        Rotation of the model's covariance matrix, increases counterclockwise.
        In radians.
    """
    mode = "progressive"
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
    """A catalog of moving objects.

    Assumed to be symmetric 2D Gaussians whose centroids are defined in pixel
    space and moving in linear fashion with velocity also defined in pixel space.
    The units are relative to the timestep.
    """
    default_config = ObjectCatalogConfig

    def __init__(self, table, **kwargs):
        # Obj cat always has to return a copy
        kwargs["return_copy"] = True
        super().__init__(table=table, **kwargs)
