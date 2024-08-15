import random

from astropy.time import Time
import astropy.units as u

__all__ = [
    "IncrementObstime",
    "ObstimeIterator",
]


class IncrementObstime:
    default_unit = "day"
    def __init__(self, start, dt):
        self.start = Time(start)
        if not isinstance(dt, u.Quantity):
            dt = dt * getattr(u, self.default_unit)
        self.dt = dt

    def __call__(self, header_val):
        curr = self.start
        self.start += self.dt
        return curr.fits


class ObstimeIterator:
    def __init__(self, obstimes, **kwargs):
        self.obstimes = Time(obstimes, **kwargs)
        self.generator = (t for t in obstimes)

    def __call__(self, header_val):
        return Time(next(self.generator)).fits


class DitherValue:
    def __init__(self, value, dither_range):
        self.value = value
        self.dither_range = dither_range

    def __call__(self, header_val):
        return self.value + random.uniform(self.dither_range)

