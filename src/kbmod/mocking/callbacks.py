__all__ = [
    "IncrementObstime",
    "ObstimeIterator",
]

class IncrementObstime:
    def __init__(self, start, dt):
        self.start = start
        self.dt = dt

    def __call__(self, mut_val):
        curr = self.start
        self.start += self.dt
        return curr

class ObstimeIterator:
    def __init__(self, obstimes):
        self.obstimes = obstimes
        self.generator  = (t for t in obstimes)

    def __call__(self, mut_val):
        return next(self.generator)

