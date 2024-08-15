import copy

__all__ = ["Config", "ConfigurationError"]


class ConfigurationError(Exception):
    """Error that is raised when configuration parameters contain a logical error."""


class Config:
    """Base class for Standardizer configuration.

    Not all standardizers will (can) use the same parameters so refer to their
    respective documentation for a more complete list.

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
        if config is not None:
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
        """Update this config from dict/other config/iterable.

        A dict-like update. If ``conf`` is present and has a ``.keys()``
        method, then does:  ``for k in conf: this[k] = conf[k]``. If ``conf``
        is present but lacks a ``.keys()`` method, then does:
        ``for k, v in conf: this[k] = v``.

        In either case, this is followed by:
        ``for k in kwargs:  this[k] = kwargs[k]``
        """
        if conf is not None:
            self._conf.update(conf)
        self._conf.update(kwargs)

    def toDict(self):
        """Return this config as a dict."""
        return self._conf

