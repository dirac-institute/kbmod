import copy

__all__ = ["Config", "ConfigurationError"]


class ConfigurationError(Exception):
    """Error that is raised when configuration parameters contain a logical error."""


class Config:
    """Base configuration class.

    Config classes that inherit from this class define configuration as their
    class attributes. Particular attributes can be overriden on an per-instance
    basis by providing a config overrides at initialization time.

    Configs inheriting from this config support basic dictionary operations.

    Parameters
    ----------
    config : `dict`, `Config` or `None`, optional
        Collection of configuration key-value pairs.
    kwargs : optional
        Keyword arguments, assigned as configuration key-values.
    """

    def __init__(self, config=None, method="default", **kwargs):
        # This is a bit hacky, but it makes life a lot easier because it
        # enables automatic loading of the default configuration and separation
        # of default config from instance bound config
        keys = list(set(dir(self.__class__)) - set(dir(Config)))

        # First fill out all the defaults by copying cls attrs
        self._conf = {k: copy.copy(getattr(self, k)) for k in keys}

        # Then override with any user-specified values
        self.update(config=config, method=method, **kwargs)

    @classmethod
    def from_configs(cls, *args):
        config = cls()
        for conf in args:
            config.update(config=conf, method="extend")
        return config

    def __getitem__(self, key):
        return self._conf[key]

    # now just shortcut the most common dict operations
    def __getattribute__(self, key):
        hasconf = "_conf" in object.__getattribute__(self, "__dict__")
        if hasconf:
            conf = object.__getattribute__(self, "_conf")
            if key in conf:
                return conf[key]
        return object.__getattribute__(self, key)

    def __setitem__(self, key, value):
        self._conf[key] = value

    def __repr__(self):
        res = f"{self.__class__.__name__}("
        for k, v in self.items():
            res += f"{k}: {v}, "
        return res[:-2] + ")"

    def __str__(self):
        res = f"{self.__class__.__name__}("
        for k, v in self.items():
            res += f"{k}: {v}, "
        return res[:-2] + ")"

    def _repr_html_(self):
        repr = f"""
        <table style='tr:nth-child(even){{background-color: #dddddd;}};'>
        <caption>{self.__class__.__name__}</caption>
          <tr>
            <th>Key</th>
            <th>Value</th>
          </tr>
        """
        for k, v in self.items():
            repr += f"<tr><td>{k}</td><td>{v}\n"
        repr += "</table>"
        return repr

    def __len__(self):
        return len(self._conf)

    def __contains__(self, key):
        return key in self._conf

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._conf == other._conf
        elif isinstance(other, dict):
            return self._conf == other
        else:
            return super().__eq__(other)

    def __iter__(self):
        return iter(self._conf)

    def __or__(self, other):
        if isinstance(other, type(self)):
            return self.__class__(config=other._conf | self._conf)
        elif isinstance(other, dict):
            return self.__class__(config=self._conf | other)
        else:
            raise TypeError("unsupported operand type(s) for |: {type(self)}and {type(other)}")

    def keys(self):
        """A set-like object providing a view on config's keys."""
        return self._conf.keys()

    def values(self):
        """A set-like object providing a view on config's values."""
        return self._conf.values()

    def items(self):
        """A set-like object providing a view on config's items."""
        return self._conf.items()

    def copy(self):
        return self.__class__(config=self._conf.copy())

    def update(self, config=None, method="default", **kwargs):
        """Update this config from dict/other config/iterable and
        apply any explicit keyword overrides.

        A dict-like update. If ``conf`` is given and has a ``.keys()``
        method, performs:

            for k in conf: this[k] = conf[k]

        If ``conf`` is given but lacks a ``.keys()`` method, performs:

            for k, v in conf: this[k] = v

        In both cases, explicit overrides are applied at the end:

            for k in kwargs:  this[k] = kwargs[k]
        """
        # Python < 3.9 does not support set operations for dicts
        # [fixme]: Update this to: other = conf | kwargs
        # and remove current implementation when 3.9 gets too old. Order of
        # conf and kwargs matter to correctly apply explicit overrides

        # Check if both conf and kwargs are given, just conf or just
        # kwargs. If none are given do nothing to comply with default
        # dict behavior
        if config is not None and kwargs:
            other = {**config, **kwargs}
        elif config is not None:
            other = config
        elif kwargs is not None:
            other = kwargs
        else:
            return

        # then, see if we the given config and overrides are a subset of this
        # config or it's superset. Depending on the selected method then raise
        # errors, ignore or extend the current config if the given config is a
        # superset (or disjoint) from the current one.
        subset = {k: v for k, v in other.items() if k in self._conf}
        superset = {k: v for k, v in other.items() if k not in subset}

        if method.lower() == "default":
            if superset:
                raise ConfigurationError(
                    "Tried setting the following fields, not a part of "
                    f"this configuration options: {superset}"
                )
            conf = other  # == subset
        elif method.lower() == "subset":
            conf = subset
        elif method.lower() == "extend":
            conf = other
        else:
            raise ValueError(
                "Method expected to be one of 'default', " f"'subset'  or 'extend'. Got {method} instead."
            )

        self._conf.update(conf)

    def toDict(self):
        """Return this config as a dict."""
        return self._conf
