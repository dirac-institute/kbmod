import random

from astropy.time import Time
import astropy.units as u


__all__ = ["IncrementObstime", "ObstimeIterator"]


class IncrementObstime:
    """Endlessly incrementing FITS-standard timestamp.

    Parameters
    ----------
    start : `astropy.time.Time`
        Starting timestamp, or a value from which AstroPy can instantiate one.
    dt : `float` or `astropy.units.Quantity`
        Size of time-step to take. Assumed to be in days by default.

    Examples
    --------
    >>> from kbmod.mocking import IncrementObstime
    >>> obst = IncrementObstime("2021-01-01T00:00:00.0000", 1)
    >>> obst()
    '2021-01-01T00:00:00.000'
    >>> obst()
    '2021-01-02T00:00:00.000'
    >>> import astropy.units as u
    >>> obst = IncrementObstime("2021-01-01T00:00:00.0000", 1*u.hour)
    >>> obst(); obst()
    '2021-01-01T00:00:00.000'
    '2021-01-01T01:00:00.000'
    """

    default_unit = "day"

    def __init__(self, start, dt):
        self.start = Time(start)
        if not isinstance(dt, u.Quantity):
            dt = dt * getattr(u, self.default_unit)
        self.dt = dt

    def __call__(self, header=None):
        curr = self.start
        self.start += self.dt
        return curr.fits


class ObstimeIterator:
    """Iterate through given timestamps.

    Parameters
    ----------
    obstimes : `astropy.time.Time`
        Starting timestamp, or a value from which AstroPy can instantiate one.

    Raises
    ------
    StopIteration
        When all the obstimes are exhausted.

    Examples
    --------
    >>> from astropy.time import Time
    >>> times = Time(range(60310, 60313, 1), format="mjd")
    >>> from kbmod.mocking import ObstimeIterator
    >>> obst = ObstimeIterator(times)
    >>> obst(); obst(); obst(); obst()
    '2024-01-01T00:00:00.000'
    '2024-01-02T00:00:00.000'
    '2024-01-03T00:00:00.000'
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/local/tmp/kbmod/src/kbmod/mocking/callbacks.py", line 49, in __call__

    StopIteration
    """

    def __init__(self, obstimes, **kwargs):
        self.obstimes = Time(obstimes, **kwargs)
        self.generator = (t for t in obstimes)

    def __call__(self, header=None):
        return Time(next(self.generator)).fits
