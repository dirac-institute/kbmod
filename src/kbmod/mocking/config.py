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

    def __init__(self, config=None, **kwargs):
        # This is a bit hacky, but it makes life a lot easier because it
        # enables automatic loading of the default configuration and separation
        # of default config from instance bound config
        keys = list(set(dir(self.__class__)) - set(dir(Config)))

        # First fill out all the defaults by copying cls attrs
        self._conf = {k: getattr(self, k) for k in keys}

        # Then override with any user-specified values
        conf = config
        if isinstance(config, Config):
            conf = config._conf

        if conf is not None:
            self._conf.update(config)
        self._conf.update(kwargs)

    # now just shortcut the most common dict operations
    def __getitem__(self, key):
        return self._conf[key]

    def __setitem__(self, key, value):
        self._conf[key] = value

    def __str__(self):
        res = f"{self.__class__.__name__}("
        for k, v in self.items():
            res += f"{k}: {v}, "
        return res[:-2] + ")"

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
            raise TypeError("unsupported operand type(s) for |: {type(self)} " "and {type(other)}")

    def keys(self):
        """A set-like object providing a view on config's keys."""
        return self._conf.keys()

    def values(self):
        """A set-like object providing a view on config's values."""
        return self._conf.values()

    def items(self):
        """A set-like object providing a view on config's items."""
        return self._conf.items()

    def update(self, conf=None, **kwargs):
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
        if conf is not None:
            self._conf.update(conf)
        self._conf.update(kwargs)

    def toDict(self):
        """Return this config as a dict."""
        return self._conf
